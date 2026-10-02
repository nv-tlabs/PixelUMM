const error = document.getElementById('gallery-error');
const viewer = document.getElementById('viewer');
const viewerVideo = document.getElementById('viewer-video');
// Autoplay (muted) while on screen, pause off screen; load sources lazily.
const playObserver = new IntersectionObserver((entries) => {
  for (const { target, isIntersecting } of entries) {
    if (isIntersecting) {
      if (!target.src && target.dataset.src) target.src = target.dataset.src;
      target.play().catch(() => {});
    } else {
      target.pause();
    }
  }
}, { threshold: 0 });

playObserver.observe(document.getElementById('overview-video'));

function openViewer(item, section) {
  viewer.style.setProperty('--tile-aspect', section?.style.getPropertyValue('--tile-aspect') || '832 / 464');
  viewer.classList.toggle('viewer-small', section?.id === 't2v-192');
  viewerVideo.poster = item.poster || '';
  viewerVideo.src = item.video;
  document.getElementById('viewer-title').textContent = item.title;
  document.getElementById('viewer-meta').textContent = item.sampler;
  document.getElementById('viewer-prompt').textContent = item.prompt;
  const note = document.getElementById('viewer-note');
  note.textContent = item.note || '';
  note.hidden = !item.note;
  viewer.showModal();
  viewerVideo.play().catch(() => {});
}

viewer.addEventListener('close', () => {
  viewerVideo.pause();
  viewerVideo.removeAttribute('src');
  viewerVideo.load();
});
viewer.addEventListener('click', (event) => {
  if (event.target === viewer) viewer.close();
});

function renderSection(section) {
  const wrap = document.createElement('section');
  wrap.className = 'section';
  wrap.id = section.id;
  wrap.setAttribute('aria-labelledby', `${section.id}-title`);
  wrap.style.setProperty('--tile-aspect', section.aspect);

  const head = document.createElement('div');
  head.className = 'section-head';
  const heading = document.createElement('h2');
  heading.id = `${section.id}-title`;
  heading.textContent = section.title;
  head.append(heading);
  if (section.meta) {
    const meta = document.createElement('p');
    meta.className = 'section-meta';
    meta.textContent = section.meta;
    head.append(meta);
  }

  const gallery = document.createElement('div');
  gallery.className = `gallery gallery-${section.id}`;
  for (const item of section.examples) gallery.append(renderTile(item));

  wrap.append(head, gallery);
  return wrap;
}

function renderTile(item) {
  const tile = document.createElement('figure');
  tile.className = 'tile';

  const media = document.createElement('button');
  media.type = 'button';
  media.className = 'tile-media';
  media.setAttribute('aria-label', `Open “${item.title}”`);

  const video = document.createElement('video');
  video.dataset.src = item.video;
  if (item.poster) video.poster = item.poster;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = 'none';
  video.setAttribute('aria-hidden', 'true');

  const prompt = document.createElement('span');
  prompt.className = 'tile-prompt';
  prompt.textContent = item.prompt;

  media.append(video, prompt);
  media.addEventListener('click', () => openViewer(item, media.closest('.section')));

  const caption = document.createElement('figcaption');
  caption.className = 'tile-title';
  caption.textContent = item.title;
  if (item.note) {
    const note = document.createElement('span');
    note.className = 'tile-note';
    note.textContent = item.note;
    caption.append(note);
  }

  tile.append(media, caption);
  playObserver.observe(video);
  return tile;
}

try {
  const response = await fetch('./examples.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Examples HTTP ${response.status}`);
  const { sections } = await response.json();
  const container = document.getElementById('examples');
  for (const section of sections) {
    // A section with a mount renders its tiles into a static block of the page.
    const mount = section.mount && document.getElementById(section.mount);
    if (mount) {
      mount.closest('.section').style.setProperty('--tile-aspect', section.aspect);
      for (const item of section.examples) mount.append(renderTile(item));
    } else {
      container.append(renderSection(section));
    }
  }
} catch (cause) {
  console.error(cause);
  error.hidden = false;
}
