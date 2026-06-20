/* Application state and internationalization manager */

const app = {
    currentLang: 'en',
    currentTheme: 'light',
    currentMode: null, // 'login' or 'register'
    videoStream: null,
    mediaRecorder: null,
    recordedChunks: [],
    uploadedFile: null
};

// Internationalization translations
const i18n = {
    en: {
        'app-title': 'Biometric Authentication',
        'theme': 'Theme',
        'auth-title': 'Welcome',
        'auth-subtitle': 'Please login or register to continue',
        'student-id': 'Student ID',
        'full-name': 'Full Name',
        'login': 'Login',
        'register': 'Register',
        'face-title': 'Face Recognition',
        'face-subtitle': 'Capture or upload video for authentication',
        'video-placeholder': '📹 Camera preview will appear here',
        'upload-text': 'Click to upload video or use camera',
        'use-camera': 'Use Camera',
        'submit': 'Submit',
        'back': 'Back',
        'error-student-id': 'Please enter student ID',
        'error-full-name': 'Please enter full name for registration',
        'error-no-video': 'Please capture or upload a video',
        'success-register': 'Registration successful! Welcome',
        'success-login': 'Login successful! Welcome back',
        'error-server': 'Server error. Please try again'
    },
    fa: {
        'app-title': 'احراز هویت بیومتریک',
        'theme': 'تم',
        'auth-title': 'خوش آمدید',
        'auth-subtitle': 'لطفا وارد شوید یا ثبت نام کنید',
        'student-id': 'کد دانشجویی',
        'full-name': 'نام کامل',
        'login': 'ورود',
        'register': 'ثبت نام',
        'face-title': 'تشخیص چهره',
        'face-subtitle': 'ویدیو ضبط کنید یا آپلود کنید',
        'video-placeholder': '📹 پیش‌نمایش دوربین اینجا نمایش داده می‌شود',
        'upload-text': 'برای آپلود ویدیو یا استفاده از دوربین کلیک کنید',
        'use-camera': 'استفاده از دوربین',
        'submit': 'ارسال',
        'back': 'بازگشت',
        'error-student-id': 'لطفا کد دانشجویی را وارد کنید',
        'error-full-name': 'لطفا نام کامل را برای ثبت نام وارد کنید',
        'error-no-video': 'لطفا ویدیو ضبط یا آپلود کنید',
        'success-register': 'ثبت نام موفق! خوش آمدید',
        'success-login': 'ورود موفق! خوش آمدید',
        'error-server': 'خطای سرور. لطفا دوباره تلاش کنید'
    }
};

// DOM elements
const elements = {
    authPage: document.getElementById('auth-page'),
    facePage: document.getElementById('face-page'),
    studentId: document.getElementById('student-id'),
    fullName: document.getElementById('full-name'),
    loginBtn: document.getElementById('login-btn'),
    registerBtn: document.getElementById('register-btn'),
    videoPreview: document.getElementById('video-preview'),
    videoPlaceholder: document.getElementById('video-placeholder'),
    fileUploadArea: document.getElementById('file-upload-area'),
    videoFile: document.getElementById('video-file'),
    cameraBtn: document.getElementById('camera-btn'),
    submitBtn: document.getElementById('submit-btn'),
    backBtn: document.getElementById('back-btn'),
    statusMessage: document.getElementById('status-message'),
    loadingSpinner: document.getElementById('loading-spinner'),
    themeToggle: document.getElementById('theme-toggle'),
    langToggle: document.getElementById('lang-toggle'),
    themeIcon: document.getElementById('theme-icon'),
    langText: document.getElementById('lang-text')
};

// Initialize application
function init() {
    setupEventListeners();
    loadPreferences();
    updateI18n();
}

// Event listeners setup
function setupEventListeners() {
    elements.loginBtn.addEventListener('click', () => handleAuth('login'));
    elements.registerBtn.addEventListener('click', () => handleAuth('register'));
    elements.cameraBtn.addEventListener('click', startCamera);
    elements.submitBtn.addEventListener('click', submitVideo);
    elements.backBtn.addEventListener('click', goBack);
    elements.themeToggle.addEventListener('click', toggleTheme);
    elements.langToggle.addEventListener('click', toggleLanguage);
    elements.fileUploadArea.addEventListener('click', () => elements.videoFile.click());
    elements.videoFile.addEventListener('change', handleFileUpload);
}

// Load user preferences from localStorage
function loadPreferences() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const savedLang = localStorage.getItem('lang') || 'en';
    
    app.currentTheme = savedTheme;
    app.currentLang = savedLang;
    
    document.documentElement.setAttribute('data-theme', savedTheme);
    document.documentElement.setAttribute('lang', savedLang);
    document.documentElement.setAttribute('dir', savedLang === 'fa' ? 'rtl' : 'ltr');
    
    elements.themeIcon.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    elements.langText.textContent = savedLang === 'fa' ? 'FA' : 'EN';
}

// Toggle theme
function toggleTheme() {
    app.currentTheme = app.currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', app.currentTheme);
    elements.themeIcon.textContent = app.currentTheme === 'dark' ? '☀️' : '🌙';
    localStorage.setItem('theme', app.currentTheme);
}

// Toggle language
function toggleLanguage() {
    app.currentLang = app.currentLang === 'en' ? 'fa' : 'en';
    document.documentElement.setAttribute('lang', app.currentLang);
    document.documentElement.setAttribute('dir', app.currentLang === 'fa' ? 'rtl' : 'ltr');
    elements.langText.textContent = app.currentLang === 'fa' ? 'FA' : 'EN';
    localStorage.setItem('lang', app.currentLang);
    updateI18n();
}

// Update all translatable elements
function updateI18n() {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (i18n[app.currentLang][key]) {
            element.textContent = i18n[app.currentLang][key];
        }
    });

    // Update placeholders
    elements.studentId.placeholder = app.currentLang === 'en' 
        ? 'Enter your student ID' 
        : 'کد دانشجویی خود را وارد کنید';
    elements.fullName.placeholder = app.currentLang === 'en' 
        ? 'Enter your full name' 
        : 'نام کامل خود را وارد کنید';
}

// Handle authentication (login/register)
function handleAuth(mode) {
    const studentId = elements.studentId.value.trim();
    const fullName = elements.fullName.value.trim();

    if (!studentId) {
        showStatus(i18n[app.currentLang]['error-student-id'], 'error');
        return;
    }

    if (mode === 'register' && !fullName) {
        showStatus(i18n[app.currentLang]['error-full-name'], 'error');
        return;
    }

    app.currentMode = mode;
    app.studentId = studentId;
    app.fullName = fullName;

    showFacePage();
}

// Show face recognition page
function showFacePage() {
    elements.authPage.classList.add('hidden');
    elements.facePage.classList.remove('hidden');
    resetFacePage();
}

// Go back to auth page
function goBack() {
    stopCamera();
    elements.facePage.classList.add('hidden');
    elements.authPage.classList.remove('hidden');
    app.uploadedFile = null;
}

// Reset face page
function resetFacePage() {
    elements.videoPreview.classList.add('hidden');
    elements.videoPlaceholder.classList.remove('hidden');
    elements.statusMessage.style.display = 'none';
    app.recordedChunks = [];
    app.uploadedFile = null;
}

// Start camera for recording
async function startCamera() {
    try {
        stopCamera();
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 15 }
            }, 
            audio: false 
        });
        
        app.videoStream = stream;
        elements.videoPreview.srcObject = stream;
        elements.videoPreview.classList.remove('hidden');
        elements.videoPlaceholder.classList.add('hidden');

        // Start recording
        app.recordedChunks = [];
        app.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'video/webm;codecs=vp9'
        });

        app.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                app.recordedChunks.push(event.data);
            }
        };

        app.mediaRecorder.start();

        // Auto-stop after 5 seconds
        setTimeout(() => {
            if (app.mediaRecorder && app.mediaRecorder.state === 'recording') {
                app.mediaRecorder.stop();
            }
        }, 5000);

    } catch (error) {
        console.error('Camera error:', error);
        showStatus('Cannot access camera', 'error');
    }
}

// Stop camera stream
function stopCamera() {
    if (app.videoStream) {
        app.videoStream.getTracks().forEach(track => track.stop());
        app.videoStream = null;
    }
    if (app.mediaRecorder && app.mediaRecorder.state === 'recording') {
        app.mediaRecorder.stop();
    }
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
        app.uploadedFile = file;
        elements.videoPreview.src = URL.createObjectURL(file);
        elements.videoPreview.srcObject = null;
        elements.videoPreview.classList.remove('hidden');
        elements.videoPlaceholder.classList.add('hidden');
    }
}

// Process and compress video
async function processVideo(blob) {
    return new Promise((resolve) => {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(blob);
        
        video.onloadedmetadata = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set target dimensions
            const targetWidth = 360;
            const aspectRatio = video.videoHeight / video.videoWidth;
            const targetHeight = Math.round(targetWidth * aspectRatio);
            
            canvas.width = targetWidth;
            canvas.height = targetHeight;

            const stream = canvas.captureStream(15); // 15 fps
            const recorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 500000 // 500 kbps
            });

            const chunks = [];
            recorder.ondataavailable = (e) => chunks.push(e.data);
            recorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }));

            recorder.start();
            video.play();

            // Draw frames
            const drawFrame = () => {
                if (video.paused || video.ended) {
                    recorder.stop();
                    return;
                }
                ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
                requestAnimationFrame(drawFrame);
            };
            drawFrame();
        };
    });
}

// Submit video to server
async function submitVideo() {
    let videoBlob = null;

    if (app.uploadedFile) {
        videoBlob = app.uploadedFile;
    } else if (app.recordedChunks.length > 0) {
        videoBlob = new Blob(app.recordedChunks, { type: 'video/webm' });
        videoBlob = await processVideo(videoBlob);
    } else {
        showStatus(i18n[app.currentLang]['error-no-video'], 'error');
        return;
    }

    stopCamera();
    showLoading(true);

    const formData = new FormData();
    formData.append('video', videoBlob, 'video.mp4');
    formData.append('student_id', app.studentId);
    formData.append('mode', app.currentMode);
    
    if (app.currentMode === 'register') {
        formData.append('full_name', app.fullName);
    }

    try {
        const response = await fetch('/api/authenticate', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        showLoading(false);

        if (response.ok && result.success) {
            const successMsg = app.currentMode === 'register' 
                ? `${i18n[app.currentLang]['success-register']} ${result.name || app.fullName}`
                : `${i18n[app.currentLang]['success-login']} ${result.name || ''}`;
            showStatus(successMsg, 'success');
        } else {
            showStatus(result.message || i18n[app.currentLang]['error-server'], 'error');
        }
    } catch (error) {
        console.error('Submit error:', error);
        showLoading(false);
        showStatus(i18n[app.currentLang]['error-server'], 'error');
    }
}

// Show status message
function showStatus(message, type) {
    elements.statusMessage.textContent = message;
    elements.statusMessage.className = `status-message status-${type}`;
    elements.statusMessage.style.display = 'block';
    
    setTimeout(() => {
        elements.statusMessage.style.display = 'none';
    }, 5000);
}

// Show/hide loading spinner
function showLoading(show) {
    elements.loadingSpinner.style.display = show ? 'block' : 'none';
    elements.submitBtn.disabled = show;
}

// Initialize on page load
init();