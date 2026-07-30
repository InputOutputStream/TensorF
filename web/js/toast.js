// toast.js — small notification popups, bottom-right.

function toast(msg, type = 'info'){
  const box = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast' + (type === 'err' ? ' err' : type === 'ok' ? ' ok' : '');
  t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transition = 'opacity .3s';
    setTimeout(() => t.remove(), 300);
  }, 4200);
}
