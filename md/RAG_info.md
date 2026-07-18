# توضیحات جامع پروژه برای RAG و مستندسازی

## خلاصه کلی پروژه
این پروژه یک سیستم **Multi-Task Learning (MTL)** برای تشخیص Deepfake، Anti-Spoofing و مدل‌سازی Temporal Consistency است که با استفاده از یک backbone مشترک و سه task head جداگانه پیاده‌سازی شده است.

---

## معماری کلی مدل

### مدل اصلی: `MTLModel`
مدل اصلی با نام `MTLModel` طراحی شده که شامل:
- یک **backbone مشترک** از خانواده EfficientNet که از کتابخانه `timm` بارگذاری می‌شود
- سه **task-specific head** برای سه وظیفه مختلف

**جریان داده در مدل:**
1. ورودی به شکل `(B, T, C, H, W)` دریافت می‌شود که B تعداد batch، T تعداد فریم‌های زمانی، C کانال‌ها، H و W ابعاد تصویر هستند
2. تنسور به `(B*T, C, H, W)` reshape می‌شود تا همه فریم‌ها به صورت batch پردازش شوند
3. اختیاری: **Temporal Shift Module (TSM)** اعمال می‌شود
4. feature extraction از طریق backbone
5. reshape به `(B, T, D)` که D بعد فیچر است
6. temporal pooling با میانگین‌گیری روی بعد زمانی
7. خروجی‌های سه head محاسبه می‌شوند

### Backbone
- ساخته می‌شود با:
  ```python
  timm.create_model(mc.backbone, pretrained=mc.pretrained, num_classes=0, global_pool="avg")
  ```
- نام دقیق backbone از فایل config خوانده می‌شود (`mc.backbone`)
- در docstring مدل به صراحت از **shared EfficientNet backbone** نام برده شده است
- backbone از وزن‌های pretrained استفاده می‌کند (قابل تنظیم)

---

## Task Heads

### 1. DeepfakeHead
- یک head برای **binary classification** جهت تشخیص deepfake
- معماری:
  - `Linear layer → ReLU → Dropout → Linear(1)`
- خروجی: یک logit برای تشخیص real/fake

### 2. AntiSpoofHead
- یک head برای **binary classification** جهت anti-spoofing
- معماری مشابه DeepfakeHead:
  - `Linear layer → ReLU → Dropout → Linear(1)`
- خروجی: یک logit برای تشخیص live/spoof

### 3. TemporalHead
- برای مدل‌سازی temporal consistency
- معماری:
  - `Linear → ReLU → Dropout → Linear(hidden_dim)`
  - سپس یک `Linear(hidden, 1)` برای classification
- خروجی:
  - `proj`: پروژکشن فیچرهای temporal
  - `logit`: خروجی classification

---

## Temporal Shift Module (TSM)

### کلاس `TemporalShift`
- یک ماژول lightweight برای مدل‌سازی temporal بدون افزودن پارامتر اضافی
- بخشی از کانال‌ها را در بعد زمانی shift می‌کند

### تابع `apply_tsm`
- روی تنسورهای flatten شده `(B*T, C, H, W)` عمل می‌کند
- Reshape به `(B, T, C, H, W)`
- بخشی از کانال‌ها را به جلو و عقب در زمان shift می‌دهد
- برمی‌گرداند به `(B*T, C, H, W)`

این مکانیسم به مدل اجازه می‌دهد بدون اضافه کردن پارامتر، از اطلاعات temporal استفاده کند.

---

## Loss Functions

### 1. Deepfake Loss
- **BCEWithLogitsLoss** استاندارد PyTorch
- برای binary classification بین real و fake

### 2. Anti-Spoof Loss: `FocalLoss`
- یک **Focal Loss** برای مقابله با class imbalance در anti-spoofing
- پارامترها:
  - `gamma`: فاکتور focusing
  - `alpha`: وزن کلاس
- فرمول:
  - محاسبه BCE با logits
  - `p_t = exp(-bce)`
  - اعمال `alpha_t * (1 - p_t)^gamma`
- میانگین loss برگردانده می‌شود

### 3. Temporal Consistency Loss
- تابع `temporal_consistency_loss`
- از **cosine similarity** بین پروژکشن‌های temporal فریم‌های مجاور استفاده می‌کند
- برای ویدیوهای واقعی: similarity را تشویق می‌کند
- برای ویدیوهای جعلی: dissimilarity را تشویق می‌کند

---

## مکانیزم‌های بهینه‌سازی چندوظیفه‌ای

### GradNorm
کلاس `GradNormManager` برای **تعادل خودکار وزن‌های task** طراحی شده است:
- loss اولیه هر task را ذخیره می‌کند
- gradient norm هر task را محاسبه می‌کند
- relative inverse training rate محاسبه می‌شود
- `model.log_weights` به‌روزرسانی می‌شود
- فعال‌سازی از طریق `cfg.train.use_gradnorm`

**مکانیزم وزن‌دهی:**
- `self.log_weights = nn.Parameter(torch.zeros(3))`
- تبدیل به وزن‌های task با `softmax`
- این وزن‌ها در صورت فعال بودن GradNorm به کار می‌روند

### PCGrad
- تابع `pcgrad_step(grads)` در کد موجود است
- در بخش optimizer-step به عنوان گزینه جایگزین GradNorm ذکر شده است
- PCGrad gradient conflictها را حل می‌کند با project کردن gradientهای متضاد

---

## داده‌ها و دیتاست‌ها

### دیتاست‌های استفاده شده
1. **FaceForensics++ (FF++)**
   - برای deepfake detection
   - شامل ویدیوهای واقعی و جعلی با compression ratioهای مختلف

2. **SiW-Mv2**
   - برای anti-spoofing
   - شامل نمونه‌های live و spoof

### استراتژی نمونه‌برداری
- training loader از FF++ و SiW-Mv2 به صورت **interleaved** استفاده می‌کند
- از `WeightedRandomSampler` برای کنترل نسبت نمونه‌ها استفاده می‌شود
- نسبت کنترل می‌شود با `cfg.train.ff_sample_ratio`
- این تضمین می‌کند که در هر batch هر دو دیتاست حضور داشته باشند

---

## فرآیند آموزش

### راه‌اندازی Trainer
- **مدل:** `MTLModel(cfg).to(device)`
- **Loss functions:**
  - `nn.BCEWithLogitsLoss()` برای deepfake
  - `FocalLoss(gamma, alpha)` برای anti-spoof
- **Optimizer:** `AdamW` با parameter groups جداگانه برای heads و backbone
- **Scheduler:** گزینه‌های cosine / step / ReduceLROnPlateau
- **AMP:** `GradScaler` برای mixed precision training

### مرحله Training Step
تابع `_train_step`:
1. دریافت `frames`, `label`, `task`
2. forward pass تحت autocast (برای AMP)
3. محاسبه سه loss:
   - `loss_df` (deepfake)
   - `loss_sp` (spoof)
   - `loss_temp` (temporal)
4. stack کردن lossها
5. اعمال وزن‌ها:
   - یا وزن‌های GradNorm
   - یا وزن‌های ثابت از config
6. total loss تقسیم بر `grad_accum_steps`
7. backward از طریق `scaler.scale(total_loss).backward()`

### مرحله Optimizer Step
تابع `_optimizer_step`:
1. به‌روزرسانی اختیاری GradNorm
2. `scaler.unscale_(optimizer)`
3. gradient clipping با `clip_grad_norm_`
4. `scaler.step(optimizer)`
5. `scaler.update()`
6. zero کردن gradients

### حلقه Epoch
تابع `train_epoch`:
- `model.train()`
- حلقه روی batchها
- انجام gradient accumulation
- مدیریت OOM errors:
  - پاک کردن CUDA cache
  - کاهش batch size
  - fallback به CPU
- برگرداندن میانگین متریک‌ها

---

## ارزیابی و متریک‌ها

### متریک‌های Deepfake
- **AUC** (Area Under Curve)
- **AP** (Average Precision)
- **EER** (Equal Error Rate)
- **Accuracy** در بهترین threshold
- **Best threshold**
- اختیاری: `video_auc`
- اختیاری: AUC برای compression ratioهای مختلف

### متریک‌های Anti-Spoofing
- **APCER** (Attack Presentation Classification Error Rate)
- **BPCER** (Bona Fide Presentation Classification Error Rate)
- **ACER** (Average Classification Error Rate)
- **HTER** (Half Total Error Rate)
- **TPR@FPR=1%** (True Positive Rate at 1% False Positive Rate)
- **AUC**

### جریان Evaluation
1. FF++ loader → محاسبه متریک‌های deepfake
2. SiW-Mv2 loader → محاسبه متریک‌های spoof
3. Temporal head روی هر دو loader ارزیابی می‌شود

---

## تنظیمات و Configuration

### Model Config
- `mc.backbone`: نام backbone از timm
- `mc.pretrained`: استفاده از وزن‌های pretrained
- `mc.use_tsm`: فعال‌سازی TSM
- `mc.tsm_shift_ratio`: نسبت shift در TSM

### Training Config
- `cfg.train.num_frames`: تعداد فریم‌ها در هر کلیپ
- `cfg.train.ff_sample_ratio`: نسبت نمونه‌برداری از FF++
- `cfg.train.use_amp`: فعال‌سازی mixed precision
- `cfg.train.use_gradnorm`: فعال‌سازی GradNorm
- `cfg.train.focal_gamma` و `focal_alpha`: پارامترهای Focal Loss

---

## نکات فنی و محدودیت‌ها

**موارد تأیید شده:**
- Backbone از `timm` و config-driven
- TSM برای temporal modeling
- Multi-task learning با سه head
- GradNorm و PCGrad برای task balancing
- Interleaved dataset training
- AMP و gradient accumulation
- Focal Loss برای class imbalance

## آموزش MTL برای سه Head روی AceFForensics++ و SiW-Mv2

### پاراگراف ۱: ساختار داده و تعریف Task ها

دو دیتاست AceFForensics++ و SiW-Mv2 از نظر domain و label space کاملاً ناهمگن هستند. AceFForensics++ یک dataset forgery detection است که ویدیوهای جعلی را با روش‌های مختلف مانند FaceSwap، Face2Face، NeuralTextures و DeepFakes تولید کرده و label های binary (real/fake) در سطح کلیپ ارائه می‌دهد. SiW-Mv2 یک dataset anti-spoofing است که شامل انواع حملات فیزیکی مانند print attack، replay attack، 3D mask، partial attack و makeup attack است و label های multi-class spoof type را در اختیار می‌گذارد. در یک pipeline یکپارچه MTL، هر batch باید از هر دو dataset به‌صورت interleaved sampling تشکیل شود — نه concatenation ساده — زیرا در غیر این صورت مدل در طول epoch هایی که فقط یک dataset می‌بیند، به task overfitting می‌کند. برای head سوم یعنی temporal consistency، label مستقیمی در این دیتاست‌ها وجود ندارد و باید به‌صورت **self-supervised** از طریق consistency loss بین فریم‌های مجاور ساخته شود: اگر فریم‌های یک کلیپ real کاملاً consistent باشند، temporal loss آن‌ها باید کمینه باشد، در حالی که در deepfake ها flickering artifact های بین‌فریمی این consistency را می‌شکنند. بنابراین، temporal head نیاز به sampling چندین فریم به‌صورت $T \in \{4, 8\}$ از یک کلیپ دارد که فاصله زمانی بین آن‌ها به‌صورت random jitter انتخاب می‌شود تا مدل به temporal stride خاصی overfit نکند.

---

### پاراگراف ۲: معماری، Loss Formulation و Gradient Management

backbone مشترک — معمولاً EfficientNet-B2 یا ViT-Small — ویژگی‌های مشترک را استخراج می‌کند و سه head مجزا روی آن سوار می‌شوند. تابع loss کلی به صورت زیر تعریف می‌شود:

$$\mathcal{L}_{total} = w_1 \mathcal{L}_{deepfake} + w_2 \mathcal{L}_{spoof} + w_3 \mathcal{L}_{temporal}$$

که در آن $\mathcal{L}_{deepfake}$ یک Binary Cross-Entropy روی AceFForensics++ است، $\mathcal{L}_{spoof}$ یک Focal Loss روی SiW-Mv2 است (به دلیل class imbalance شدید بین انواع attack)، و $\mathcal{L}_{temporal}$ یک Cosine Embedding Loss یا MSE بین feature vector های فریم‌های مجاور در همان کلیپ است. وزن‌های $w_i$ نباید ثابت باشند — **GradNorm** در هر iteration این وزن‌ها را بر اساس نرم گرادیان هر task نسبت به یک target rate تنظیم می‌کند:

$$\hat{w}_i(t) \leftarrow \hat{w}_i(t) \cdot \frac{N \cdot \bar{g}(t)}{G_i(t)}$$

که $G_i(t)$ نرم گرادیان task $i$ در step $t$ و $\bar{g}(t)$ میانگین آن‌هاست. اما مشکل اساسی اینجاست که GradNorm فقط magnitude را تنظیم می‌کند، نه جهت. برای مقابله با gradient conflict، باید از **PCGrad** یا **CAGrad** استفاده کرد: در PCGrad اگر گرادیان دو task زاویه‌ای بیش از ۹۰ درجه داشته باشند (یعنی $\cos\theta_{ij} < 0$)، component متعارض از گرادیان task اول روی جهت task دوم project و حذف می‌شود تا گرادیان نهایی هر task فقط در جهت‌هایی به‌روزرسانی شود که برای task دیگر مخرب نیست.

---

### پاراگراف ۳: Training Protocol، Domain Adaptation و ارزیابی

از آنجا که AceFForensics++ و SiW-Mv2 از نظر domain کاملاً متفاوت هستند — یکی عمدتاً فشرده‌سازی ویدیویی با H.264 و دیگری تصویر چهره در شرایط محیطی متنوع — backbone باید قبلاً روی یک dataset چهره عمومی مانند VGGFace2 یا MS-Celeb-1M به‌صورت supervised pretrain شده باشد تا feature های اولیه چهره را داشته باشد، سپس fine-tuning چند مرحله‌ای انجام شود: ابتدا فقط head ها آموزش می‌بینند (backbone frozen)، سپس در مرحله دوم با learning rate بسیار پایین‌تر به صورت $lr_{backbone} = 0.1 \times lr_{heads}$ کل شبکه fine-tune می‌شود. برای ارزیابی، هر head باید جداگانه روی test split متناسب خودش سنجیده شود: deepfake head با AUC و EER روی AceFForensics++ و anti-spoof head با HTER (Half Total Error Rate) روی SiW-Mv2 که metric استاندارد در ادبیات PAD است، چون HTER نرخ خطای بین APCER و BPCER را متوازن می‌کند:

$$HTER = \frac{APCER + BPCER}{2}$$

و temporal head با variance گرادیان بین‌فریمی در feature space ارزیابی می‌شود که در کلیپ‌های real باید پایین‌تر از کلیپ‌های fake باشد. در عمل، cross-dataset generalization مهم‌ترین آزمون است — مدلی که روی AceFForensics++ آموزش دیده، باید بتواند روی DFDC یا FaceShifter نیز به AUC قابل قبول ($> 0.75$) برسد، که این امر تنها با استفاده از feature های generalizable در لایه‌های میانی backbone، نه artifact های dataset-specific، ممکن است.