const form = document.getElementById('upscaleForm');
const imageInput = document.getElementById('imageInput');
const scaleSelect = document.getElementById('scaleSelect');
const enhanceRange = document.getElementById('enhanceRange');
const enhanceValue = document.getElementById('enhanceValue');
const originalPreview = document.getElementById('originalPreview');
const resultPreview = document.getElementById('resultPreview');
const resultMeta = document.getElementById('resultMeta');
const downloadLink = document.getElementById('downloadLink');
const processBtn = document.getElementById('processBtn');
const fileDrop = document.getElementById('fileDrop');
const toolSelect = document.getElementById('toolSelect');
const activeToolTitle = document.getElementById('activeToolTitle');
const activeToolDesc = document.getElementById('activeToolDesc');

const controlsUpscale = document.getElementById('controlsUpscale');
const controlsBgRemove = document.getElementById('controlsBgRemove');
const controlsCartoon = document.getElementById('controlsCartoon');
const controlsResize = document.getElementById('controlsResize');
const controlsConvert = document.getElementById('controlsConvert');

const cartoonStyle = document.getElementById('cartoonStyle');
const keepObjects = document.getElementById('keepObjects');
const resizeWidth = document.getElementById('resizeWidth');
const resizeHeight = document.getElementById('resizeHeight');
const keepAspect = document.getElementById('keepAspect');
const convertFormat = document.getElementById('convertFormat');
const toolDescription = document.getElementById('toolDescription');

function setLoading(isLoading) {
  // Show processing indicator in footer only when active
  const processBar = document.getElementById('processBar');
  if (processBar) processBar.hidden = !isLoading;
  processBtn.disabled = isLoading;
}

function updateEnhanceLabel() {
  enhanceValue.textContent = `${Number(enhanceRange.value).toFixed(1)}x`;
}

updateEnhanceLabel();
enhanceRange.addEventListener('input', updateEnhanceLabel);

fileDrop.addEventListener('click', () => imageInput.click());

// Drag and drop behavior
['dragenter', 'dragover'].forEach(evt => fileDrop.addEventListener(evt, e => {
  e.preventDefault();
  e.stopPropagation();
  fileDrop.style.borderColor = '#3f8cff';
}));

['dragleave', 'drop'].forEach(evt => fileDrop.addEventListener(evt, e => {
  e.preventDefault();
  e.stopPropagation();
  fileDrop.style.borderColor = '';
}));

fileDrop.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files && files[0]) {
    imageInput.files = files;
    previewOriginal(files[0]);
  }
});

imageInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) previewOriginal(file);
});

function previewOriginal(file) {
  const url = URL.createObjectURL(file);
  originalPreview.src = url;
}

function switchToolUI(tool) {
  // Hide all control panels first
  controlsUpscale.hidden = true;
  controlsBgRemove.hidden = true;
  controlsCartoon.hidden = true;
  controlsResize.hidden = true;
  controlsConvert.hidden = true;
  
  // Hide all input fields that are not relevant to the selected tool
  cartoonStyle.parentElement.hidden = true;
  resizeWidth.parentElement.hidden = true;
  resizeHeight.parentElement.hidden = true;
  keepAspect.parentElement.hidden = true;
  convertFormat.parentElement.hidden = true;
  scaleSelect.parentElement.hidden = true;
  enhanceRange.parentElement.hidden = true;
  
  // Hide keep objects control by default
  const keepObjectsControl = document.getElementById('keepObjectsControl');
  if (keepObjectsControl) keepObjectsControl.hidden = true;
  
  // Show only the selected tool's controls
  if (tool === 'cartoon') {
    controlsCartoon.hidden = false;
    cartoonStyle.parentElement.hidden = false;
  } else if (tool === 'upscale') {
    controlsUpscale.hidden = false;
    scaleSelect.parentElement.hidden = false;
    enhanceRange.parentElement.hidden = false;
  } else if (tool === 'bgremove') {
    controlsBgRemove.hidden = false;
    // Show keep objects control for background removal
    if (keepObjectsControl) keepObjectsControl.hidden = false;
  } else if (tool === 'resize') {
    controlsResize.hidden = false;
    resizeWidth.parentElement.hidden = false;
    resizeHeight.parentElement.hidden = false;
    keepAspect.parentElement.hidden = false;
  } else if (tool === 'convert') {
    controlsConvert.hidden = false;
    convertFormat.parentElement.hidden = false;
  }
  
  const map = {
    cartoon: { 
      title: 'AI Cartoonizer', 
      desc: 'Turn photos into cartoon, pencil, watercolor and more.',
      description: 'Transform photos into cartoon, pencil, watercolor, ink, and comic styles with AI. Perfect for creating artistic versions of your photos.'
    },
    upscale: { 
      title: 'Image Upscaler', 
      desc: 'Enhance details and upscale your images with AI.',
      description: 'Enhance image quality and upscale to 4x resolution with Real-ESRGAN AI. Great for improving low-resolution images.'
    },
    bgremove: { 
      title: 'Background Remover', 
      desc: 'Remove backgrounds to transparent PNGs.',
      description: 'Remove backgrounds with AI detection for people, faces, and objects. Creates transparent PNGs for easy editing.'
    },
    resize: { 
      title: 'Image Resizer', 
      desc: 'Resize to exact dimensions while keeping quality.',
      description: 'Resize images to exact dimensions while maintaining quality and aspect ratio. Perfect for social media and printing.'
    },
    convert: { 
      title: 'Image Converter', 
      desc: 'Convert between PNG, JPG and WEBP formats.',
      description: 'Convert between PNG, JPG, and WEBP formats with optimized settings. Choose the best format for your needs.'
    },
  };
  activeToolTitle.textContent = map[tool].title;
  activeToolDesc.textContent = map[tool].desc;
  toolDescription.innerHTML = `<p>${map[tool].description}</p>`;
  processBtn.textContent =
    tool === 'cartoon' ? 'Cartoonize' :
    tool === 'upscale' ? 'Upscale' :
    tool === 'bgremove' ? 'Remove Background' :
    tool === 'resize' ? 'Resize' :
    'Convert';
}

toolSelect.addEventListener('change', () => {
  switchToolUI(toolSelect.value);
});

switchToolUI(toolSelect.value);

// Feature cards functionality
document.querySelectorAll('.feature-card').forEach(card => {
  card.addEventListener('click', () => {
    const tool = card.getAttribute('data-tool');
    toolSelect.value = tool;
    switchToolUI(tool);
    
    // Scroll to upload section
    document.querySelector('.upload-panel').scrollIntoView({ 
      behavior: 'smooth',
      block: 'start'
    });
  });
});

// Navigation smooth scrolling for in-page anchors only
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    const href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      e.preventDefault();
      const targetId = href.substring(1);
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({ 
          behavior: 'smooth',
          block: 'start'
        });
      }
    }
  });
});

// Login button navigates to login page (handled by anchor)

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const file = imageInput.files?.[0];
  if (!file) {
    alert('Please select an image first.');
    return;
  }

  const tool = toolSelect.value;
  const formData = new FormData();
  formData.append('image', file);

  let endpoint = '/api/cartoonize';
  if (tool === 'cartoon') {
    formData.append('style', cartoonStyle.value);
    endpoint = '/api/cartoonize';
  } else if (tool === 'upscale') {
    formData.append('scale', scaleSelect.value);
    formData.append('enhance', enhanceRange.value);
    endpoint = '/api/upscale';
  } else if (tool === 'bgremove') {
    formData.append('keep_objects', keepObjects.value);
    endpoint = '/api/remove_background';
  } else if (tool === 'resize') {
    const w = resizeWidth.value;
    const h = resizeHeight.value;
    if (!w && !h) {
      alert('Enter width or height.');
      return;
    }
    if (w) formData.append('width', w);
    if (h) formData.append('height', h);
    formData.append('keep_aspect', keepAspect.checked ? 'true' : 'false');
    endpoint = '/api/resize';
  } else if (tool === 'convert') {
    formData.append('format', convertFormat.value);
    endpoint = '/api/convert';
  }

  try {
    setLoading(true);
    resultMeta.textContent = '';
    resultPreview.removeAttribute('src');
    downloadLink.style.display = 'none';

    const res = await fetch(endpoint, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(err.error || 'Failed to process image');
    }

    const data = await res.json();
    const url = data.output_url;
    resultPreview.src = url;
    // Build meta text based on tool
    let meta = '';
    if (tool === 'cartoon') {
      meta = `Style: ${data.style}`;
    } else if (tool === 'upscale') {
      meta = `Scale: ${data.scale}x • Enhance: ${Number(data.enhance).toFixed(1)}x`;
    } else if (tool === 'resize') {
      meta = `Output: ${data.width}×${data.height}`;
    } else if (tool === 'convert') {
      meta = `Format: ${data.format}`;
    } else if (tool === 'bgremove') {
      const kept = data.kept_objects || 'background';
      meta = `Kept: ${kept} • Method: ${data.method || 'smart'} • Transparent PNG`;
    }
    resultMeta.textContent = meta;
    downloadLink.href = url;
    const ext = (tool === 'bgremove') ? 'png' : (tool === 'convert' ? (data.format === 'jpeg' ? 'jpg' : data.format) : 'jpg');
    downloadLink.download = `photoxcel_${Date.now()}.${ext}`;
    downloadLink.style.display = 'inline-block';
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
});


