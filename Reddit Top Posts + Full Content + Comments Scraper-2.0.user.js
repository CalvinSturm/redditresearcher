// ==UserScript==
// @name         Reddit Top Posts + Full Content + Comments Scraper
// @namespace    http://tampermonkey.net/
// @version      1.9
// @description  Auto-scroll Reddit listing pages, scrape visible posts, and export listing data or full post content + comments
// @match        https://www.reddit.com/*
// @match        https://old.reddit.com/*
// @grant        GM_registerMenuCommand
// @grant        GM_setClipboard
// @connect      reddit.com
// @connect      www.reddit.com
// @connect      old.reddit.com
// ==/UserScript==

(function () {
  'use strict';

  const CONFIG = {
    autoScroll: {
      maxSteps: 60,
      delayMs: 1400,
      settleMs: 600,
      stopAfterNoGrowthSteps: 4
    },
    fullScrape: {
      delayBetweenPostsMs: 900,
      maxPosts: 300,
      retryCount: 2,
      retryDelayMs: 1200
    },
    defaults: {
      timeframe: 'week',        // day | week | month | year
      minScore: 0,
      maxScore: 0,              // 0 = unlimited
      minComments: 0,
      stopAfterNMatches: 0      // 0 = unlimited
    }
  };

  const PANEL_ID = 'tm-reddit-scraper-panel';
  const STATUS_ID = 'tm-reddit-scraper-status';
  const LOG_ID = 'tm-reddit-scraper-log';
  const RESTORE_ID = 'tm-reddit-scraper-restore';
  const STYLE_ID = 'tm-reddit-scraper-styles';

  const SETTINGS = {
    timeframe: CONFIG.defaults.timeframe,
    minScore: CONFIG.defaults.minScore,
    maxScore: CONFIG.defaults.maxScore,
    minComments: CONFIG.defaults.minComments,
    stopAfterNMatches: CONFIG.defaults.stopAfterNMatches
  };

  let isRunning = false;
  let observerStarted = false;

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function cleanText(value) {
    return String(value || '')
      .replace(/\u00A0/g, ' ')
      .replace(/[ \t\r\f\v]+/g, ' ')
      .trim();
  }

  function cleanMultilineText(value) {
    return String(value || '')
      .replace(/\u00A0/g, ' ')
      .replace(/\r/g, '')
      .split('\n')
      .map(line => cleanText(line))
      .filter(Boolean)
      .join('\n')
      .trim();
  }

  function absoluteUrl(href, base) {
    if (!href) return '';
    try {
      return new URL(href, base || location.href).toString();
    } catch {
      return '';
    }
  }

  function safeFilenamePart(value) {
    return String(value || '')
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
      .replace(/\s+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_+|_+$/g, '') || 'reddit';
  }

  function parseCompactNumber(text) {
    const raw = cleanText(text).toLowerCase();
    if (!raw) return null;

    if (/^(\d+(?:[.,]\d+)?)\s*([km])$/i.test(raw)) {
      const match = raw.match(/^(\d+(?:[.,]\d+)?)\s*([km])$/i);
      let num = parseFloat(match[1].replace(/,/g, ''));
      if (Number.isNaN(num)) return null;
      const suffix = match[2].toLowerCase();
      if (suffix === 'k') num *= 1000;
      if (suffix === 'm') num *= 1000000;
      return Math.round(num);
    }

    const match = raw.match(/(\d+(?:[.,]\d+)?)\s*([km])?/i);
    if (!match) return null;

    let num = parseFloat(match[1].replace(/,/g, ''));
    if (Number.isNaN(num)) return null;

    const suffix = (match[2] || '').toLowerCase();
    if (suffix === 'k') num *= 1000;
    if (suffix === 'm') num *= 1000000;

    return Math.round(num);
  }

  function parseScore(text) {
    const raw = cleanText(text).toLowerCase();
    if (!raw) return null;
    if (raw.includes('vote') || raw.includes('point') || /^[\d.,]+\s*[km]?$/i.test(raw)) {
      return parseCompactNumber(raw);
    }
    return null;
  }

  function parseComments(text) {
    const raw = cleanText(text).toLowerCase();
    if (!raw) return null;
    if (!raw.includes('comment')) return null;
    if (raw.includes('comment') && !/\d/.test(raw)) return 0;
    return parseCompactNumber(raw);
  }

  function isOldReddit() {
    return location.hostname === 'old.reddit.com';
  }

  function getUrlInfo(urlOverride) {
    const url = new URL(urlOverride || location.href);
    const path = url.pathname;
    const subredditMatch = path.match(/\/r\/([^/]+)/i);

    let sort = '';
    if (path.includes('/top')) sort = 'top';
    else if (path.includes('/hot')) sort = 'hot';
    else if (path.includes('/new')) sort = 'new';
    else if (path.includes('/rising')) sort = 'rising';
    else if (path.includes('/controversial')) sort = 'controversial';
    else if (url.searchParams.get('sort')) sort = url.searchParams.get('sort');

    return {
      subreddit: subredditMatch ? subredditMatch[1] : '',
      sort,
      timeframe: url.searchParams.get('t') || '',
      url: url.toString(),
      host: url.hostname,
      is_old_reddit: url.hostname === 'old.reddit.com'
    };
  }

  function normalizeCommentsPermalink(url) {
    if (!url) return '';
    try {
      const u = new URL(url, location.href);
      u.hash = '';
      u.search = '';
      return u.toString().replace(/\/$/, '');
    } catch {
      return String(url).replace(/\/$/, '');
    }
  }

  function uniqueBy(items, keyFn) {
    const seen = new Set();
    const out = [];
    for (const item of items) {
      const key = keyFn(item);
      if (!key || seen.has(key)) continue;
      seen.add(key);
      out.push(item);
    }
    return out;
  }

  function querySelectorAllSafe(root, selector) {
    try {
      return [...root.querySelectorAll(selector)];
    } catch {
      return [];
    }
  }

  function querySelectorSafe(root, selector) {
    try {
      return root.querySelector(selector);
    } catch {
      return null;
    }
  }

  function setStatus(message) {
    console.log(`[Reddit Scraper] ${message}`);
    const el = document.getElementById(STATUS_ID);
    if (el) el.textContent = message;
  }

  function appendLog(message) {
    console.log(`[Reddit Scraper] ${message}`);
    const el = document.getElementById(LOG_ID);
    if (!el) return;
    const time = new Date().toLocaleTimeString();
    el.textContent += `[${time}] ${message}\n`;
    el.scrollTop = el.scrollHeight;
  }

  function clearLog() {
    const el = document.getElementById(LOG_ID);
    if (el) el.textContent = '';
  }

  function buttonStyle(btn, tone) {
    btn.className = 'tm-rs-btn';
    btn.dataset.tone = tone;
    btn.type = 'button';
  }

  function labelStyle(el) {
    el.className = 'tm-rs-label';
  }

  function inputStyle(el) {
    el.className = 'tm-rs-input';
  }

  function selectStyle(el) {
    inputStyle(el);
    el.classList.add('tm-rs-select');
  }

  function optionStyle(el) {
    el.className = 'tm-rs-option';
  }

  function fieldWrapStyle(el, isWide) {
    el.className = `tm-rs-field${isWide ? ' tm-rs-field-wide' : ''}`;
  }

  function sectionStyle(el) {
    el.className = 'tm-rs-section';
  }

  function sectionTitleStyle(el) {
    el.className = 'tm-rs-section-title';
  }

  function helpTextStyle(el) {
    el.className = 'tm-rs-help';
  }

  function toolbarButtonStyle(btn) {
    btn.className = 'tm-rs-toolbar-btn';
    btn.type = 'button';
  }

  function ensurePanelStyles() {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      #${PANEL_ID} {
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 2147483647;
        width: min(380px, calc(100vw - 24px));
        max-height: calc(100vh - 24px);
        overflow: auto;
        box-sizing: border-box;
        padding: 16px;
        border: 1px solid rgba(255, 105, 54, 0.65);
        border-radius: 18px;
        background:
          radial-gradient(circle at top right, rgba(255, 89, 0, 0.16), transparent 32%),
          linear-gradient(180deg, rgba(21, 22, 26, 0.98), rgba(11, 12, 15, 0.98));
        color: #f5f7fb;
        font-family: "Segoe UI Variable Text", Aptos, "Trebuchet MS", sans-serif;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        pointer-events: auto;
      }

      #${PANEL_ID} * {
        box-sizing: border-box;
      }

      #${PANEL_ID}::-webkit-scrollbar,
      #${LOG_ID}::-webkit-scrollbar {
        width: 10px;
      }

      #${PANEL_ID}::-webkit-scrollbar-thumb,
      #${LOG_ID}::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.14);
        border-radius: 999px;
      }

      #${PANEL_ID} .tm-rs-header {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: flex-start;
        margin-bottom: 14px;
        cursor: move;
        user-select: none;
      }

      #${PANEL_ID} .tm-rs-title {
        margin: 0;
        font-size: 24px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.03em;
      }

      #${PANEL_ID} .tm-rs-subtitle {
        margin: 6px 0 0;
        color: #9eb4c7;
        font-size: 12px;
        line-height: 1.45;
      }

      #${PANEL_ID} .tm-rs-header-actions {
        display: flex;
        gap: 8px;
        flex-shrink: 0;
        flex-wrap: wrap;
        justify-content: flex-end;
        cursor: default;
      }

      #${PANEL_ID} .tm-rs-toolbar-btn {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        color: #f5f7fb;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
        line-height: 1;
        cursor: pointer;
        transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
      }

      #${PANEL_ID} .tm-rs-toolbar-btn:hover,
      #${PANEL_ID} .tm-rs-toolbar-btn:focus-visible {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.22);
        transform: translateY(-1px);
        outline: none;
      }

      #${PANEL_ID} .tm-rs-section {
        margin-bottom: 12px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.04);
      }

      #${PANEL_ID} .tm-rs-section-title,
      #${PANEL_ID} .tm-rs-status-label {
        margin: 0 0 6px;
        color: #d8e2ee;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      #${PANEL_ID} .tm-rs-status-text {
        font-size: 14px;
        line-height: 1.45;
        color: #f8fafc;
        word-break: break-word;
      }

      #${PANEL_ID} .tm-rs-help {
        margin: 0;
        color: #8da1b4;
        font-size: 11px;
        line-height: 1.45;
      }

      #${PANEL_ID} .tm-rs-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
      }

      #${PANEL_ID} .tm-rs-field-wide {
        grid-column: 1 / -1;
      }

      #${PANEL_ID} .tm-rs-label {
        display: block;
        margin-bottom: 6px;
        color: #c7d2df;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.25;
      }

      #${PANEL_ID} .tm-rs-input {
        width: 100%;
        min-height: 42px;
        padding: 10px 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.07);
        color: #f8fafc;
        font-size: 13px;
        line-height: 1.2;
        outline: none;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
      }

      #${PANEL_ID} .tm-rs-input::placeholder {
        color: #7f90a1;
      }

      #${PANEL_ID} .tm-rs-input:hover {
        background: rgba(255, 255, 255, 0.09);
      }

      #${PANEL_ID} .tm-rs-input:focus-visible {
        border-color: rgba(67, 160, 255, 0.75);
        box-shadow: 0 0 0 3px rgba(67, 160, 255, 0.18);
      }

      #${PANEL_ID} .tm-rs-select {
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        padding-right: 34px;
        background-image:
          linear-gradient(45deg, transparent 50%, #dbe6f3 50%),
          linear-gradient(135deg, #dbe6f3 50%, transparent 50%);
        background-position: calc(100% - 18px) calc(50% - 2px), calc(100% - 13px) calc(50% - 2px);
        background-size: 5px 5px, 5px 5px;
        background-repeat: no-repeat;
        color-scheme: dark;
      }

      #${PANEL_ID} .tm-rs-option {
        background: #16181d;
        color: #f8fafc;
      }

      #${PANEL_ID} .tm-rs-actions {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
      }

      #${PANEL_ID} .tm-rs-btn {
        width: 100%;
        min-height: 46px;
        padding: 12px 14px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 12px;
        color: #fff;
        font-size: 13px;
        font-weight: 800;
        text-align: left;
        cursor: pointer;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
        transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
      }

      #${PANEL_ID} .tm-rs-btn:hover,
      #${PANEL_ID} .tm-rs-btn:focus-visible {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.3);
        filter: brightness(1.04);
        outline: none;
      }

      #${PANEL_ID} .tm-rs-btn[data-tone="orange"] { background: linear-gradient(135deg, #ff6b2c, #ff4500); }
      #${PANEL_ID} .tm-rs-btn[data-tone="blue"] { background: linear-gradient(135deg, #2496ed, #0079d3); }
      #${PANEL_ID} .tm-rs-btn[data-tone="green"] { background: linear-gradient(135deg, #39b766, #2ea043); }
      #${PANEL_ID} .tm-rs-btn[data-tone="violet"] { background: linear-gradient(135deg, #9c73ff, #7c4dff); }

      #${PANEL_ID} .tm-rs-log {
        margin: 10px 0 0;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(0, 0, 0, 0.26);
        color: #dbe6f3;
        font-size: 11px;
        line-height: 1.45;
        max-height: 220px;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }

      #${RESTORE_ID} {
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 2147483647;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 999px;
        padding: 12px 16px;
        background: linear-gradient(135deg, #ff6b2c, #ff4500);
        color: #fff;
        font-family: "Segoe UI Variable Text", Aptos, "Trebuchet MS", sans-serif;
        font-size: 13px;
        font-weight: 800;
        cursor: pointer;
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.42);
      }

      @media (max-width: 560px) {
        #${PANEL_ID} {
          top: 12px;
          right: 12px;
          padding: 14px;
          border-radius: 16px;
        }

        #${PANEL_ID} .tm-rs-header {
          flex-direction: column;
        }

        #${PANEL_ID} .tm-rs-header-actions {
          justify-content: flex-start;
        }

        #${PANEL_ID} .tm-rs-header-actions,
        #${PANEL_ID} .tm-rs-grid,
        #${PANEL_ID} .tm-rs-actions {
          grid-template-columns: 1fr;
        }

        #${PANEL_ID} .tm-rs-field-wide {
          grid-column: auto;
        }
      }
    `;

    document.head.appendChild(style);
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function enablePanelDragging(panel, handle) {
    if (!panel || !handle) return;

    let dragState = null;

    const stopDrag = () => {
      dragState = null;
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
      document.removeEventListener('pointercancel', onPointerUp);
    };

    const onPointerMove = (event) => {
      if (!dragState) return;

      const maxLeft = Math.max(8, window.innerWidth - dragState.width - 8);
      const maxTop = Math.max(8, window.innerHeight - dragState.height - 8);
      const nextLeft = clamp(event.clientX - dragState.offsetX, 8, maxLeft);
      const nextTop = clamp(event.clientY - dragState.offsetY, 8, maxTop);

      panel.style.left = `${nextLeft}px`;
      panel.style.top = `${nextTop}px`;
      panel.style.right = 'auto';
    };

    const onPointerUp = () => {
      stopDrag();
    };

    handle.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      if (event.target.closest('button, input, select, textarea, label, a')) return;

      const rect = panel.getBoundingClientRect();
      dragState = {
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top,
        width: rect.width,
        height: rect.height
      };

      document.addEventListener('pointermove', onPointerMove);
      document.addEventListener('pointerup', onPointerUp);
      document.addEventListener('pointercancel', onPointerUp);
      event.preventDefault();
    });
  }

  function enableBubbleDragging(bubble) {
    if (!bubble) return;

    let dragState = null;
    let didDrag = false;

    const stopDrag = () => {
      dragState = null;
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
      document.removeEventListener('pointercancel', onPointerUp);
    };

    const onPointerMove = (event) => {
      if (!dragState) return;

      const nextLeft = event.clientX - dragState.offsetX;
      const nextTop = event.clientY - dragState.offsetY;
      const deltaX = nextLeft - dragState.startLeft;
      const deltaY = nextTop - dragState.startTop;

      if (!didDrag && Math.hypot(deltaX, deltaY) > 4) {
        didDrag = true;
      }

      const maxLeft = Math.max(8, window.innerWidth - dragState.width - 8);
      const maxTop = Math.max(8, window.innerHeight - dragState.height - 8);
      bubble.style.left = `${clamp(nextLeft, 8, maxLeft)}px`;
      bubble.style.top = `${clamp(nextTop, 8, maxTop)}px`;
      bubble.style.right = 'auto';
    };

    const onPointerUp = () => {
      stopDrag();
      setTimeout(() => {
        didDrag = false;
      }, 0);
    };

    bubble.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;

      const rect = bubble.getBoundingClientRect();
      dragState = {
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top,
        width: rect.width,
        height: rect.height,
        startLeft: rect.left,
        startTop: rect.top
      };

      document.addEventListener('pointermove', onPointerMove);
      document.addEventListener('pointerup', onPointerUp);
      document.addEventListener('pointercancel', onPointerUp);
      event.preventDefault();
    });

    bubble.addEventListener('click', (event) => {
      if (!didDrag) return;
      event.preventDefault();
      event.stopPropagation();
    }, true);
  }

  function createPanel() {
    if (document.getElementById(PANEL_ID)) return;
    ensurePanelStyles();

    const panel = document.createElement('div');
    panel.id = PANEL_ID;
    const header = document.createElement('div');
    header.className = 'tm-rs-header';

    const titleWrap = document.createElement('div');

    const title = document.createElement('h2');
    title.className = 'tm-rs-title';
    title.textContent = 'Reddit Scraper';

    const subtitle = document.createElement('div');
    subtitle.className = 'tm-rs-subtitle';
    subtitle.textContent = 'Cleaner filters, faster exports, and less panel clutter.';

    titleWrap.appendChild(title);
    titleWrap.appendChild(subtitle);

    const headerActions = document.createElement('div');
    headerActions.className = 'tm-rs-header-actions';

    const statusWrap = document.createElement('div');
    sectionStyle(statusWrap);
    const statusLabel = document.createElement('div');
    statusLabel.className = 'tm-rs-status-label';
    statusLabel.textContent = 'Status';

    const status = document.createElement('div');
    status.id = STATUS_ID;
    status.className = 'tm-rs-status-text';
    status.textContent = 'Ready';

    const statusHelp = document.createElement('p');
    helpTextStyle(statusHelp);
    statusHelp.textContent = 'Filters apply live while the scraper scrolls and exports.';

    statusWrap.appendChild(statusLabel);
    statusWrap.appendChild(status);
    statusWrap.appendChild(statusHelp);

    const filtersSection = document.createElement('div');
    sectionStyle(filtersSection);

    const filtersTitle = document.createElement('div');
    sectionTitleStyle(filtersTitle);
    filtersTitle.textContent = 'Filters';

    const filtersHelp = document.createElement('p');
    helpTextStyle(filtersHelp);
    filtersHelp.textContent = 'Use 0 to leave a score or limit filter open.';

    const controlsWrap = document.createElement('div');
    controlsWrap.className = 'tm-rs-grid';

    const timeframeWrap = document.createElement('div');
    fieldWrapStyle(timeframeWrap, true);

    const timeframeLabel = document.createElement('label');
    timeframeLabel.textContent = 'Timeframe';
    labelStyle(timeframeLabel);

    const timeframeSelect = document.createElement('select');
    timeframeSelect.id = 'tm-reddit-scraper-timeframe';
    selectStyle(timeframeSelect);

    [
      { value: 'day', label: '1 day' },
      { value: 'week', label: '1 week' },
      { value: 'month', label: '1 month' },
      { value: 'year', label: '1 year' }
    ].forEach(opt => {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.label;
      optionStyle(option);
      timeframeSelect.appendChild(option);
    });
    timeframeSelect.value = SETTINGS.timeframe;
    timeframeSelect.addEventListener('change', () => {
      SETTINGS.timeframe = timeframeSelect.value;
      appendLog(`Timeframe set to ${SETTINGS.timeframe}`);
    });

    timeframeWrap.appendChild(timeframeLabel);
    timeframeWrap.appendChild(timeframeSelect);

    const minScoreWrap = document.createElement('div');
    fieldWrapStyle(minScoreWrap, false);
    const minScoreLabel = document.createElement('label');
    minScoreLabel.textContent = 'Min score';
    labelStyle(minScoreLabel);

    const minScoreInput = document.createElement('input');
    minScoreInput.type = 'number';
    minScoreInput.min = '0';
    minScoreInput.step = '1';
    minScoreInput.placeholder = '0';
    minScoreInput.value = String(SETTINGS.minScore);
    minScoreInput.id = 'tm-reddit-scraper-min-score';
    inputStyle(minScoreInput);
    minScoreInput.addEventListener('input', () => {
      SETTINGS.minScore = Math.max(0, parseInt(minScoreInput.value || '0', 10) || 0);
    });

    minScoreWrap.appendChild(minScoreLabel);
    minScoreWrap.appendChild(minScoreInput);

    const maxScoreWrap = document.createElement('div');
    fieldWrapStyle(maxScoreWrap, false);
    const maxScoreLabel = document.createElement('label');
    maxScoreLabel.textContent = 'Max score';
    labelStyle(maxScoreLabel);

    const maxScoreInput = document.createElement('input');
    maxScoreInput.type = 'number';
    maxScoreInput.min = '0';
    maxScoreInput.step = '1';
    maxScoreInput.placeholder = '0';
    maxScoreInput.value = String(SETTINGS.maxScore);
    maxScoreInput.id = 'tm-reddit-scraper-max-score';
    inputStyle(maxScoreInput);
    maxScoreInput.addEventListener('input', () => {
      SETTINGS.maxScore = Math.max(0, parseInt(maxScoreInput.value || '0', 10) || 0);
    });

    maxScoreWrap.appendChild(maxScoreLabel);
    maxScoreWrap.appendChild(maxScoreInput);

    const minCommentsWrap = document.createElement('div');
    fieldWrapStyle(minCommentsWrap, false);
    const minCommentsLabel = document.createElement('label');
    minCommentsLabel.textContent = 'Min comments';
    labelStyle(minCommentsLabel);

    const minCommentsInput = document.createElement('input');
    minCommentsInput.type = 'number';
    minCommentsInput.min = '0';
    minCommentsInput.step = '1';
    minCommentsInput.placeholder = '0';
    minCommentsInput.value = String(SETTINGS.minComments);
    minCommentsInput.id = 'tm-reddit-scraper-min-comments';
    inputStyle(minCommentsInput);
    minCommentsInput.addEventListener('input', () => {
      SETTINGS.minComments = Math.max(0, parseInt(minCommentsInput.value || '0', 10) || 0);
    });

    minCommentsWrap.appendChild(minCommentsLabel);
    minCommentsWrap.appendChild(minCommentsInput);

    const stopAfterWrap = document.createElement('div');
    fieldWrapStyle(stopAfterWrap, true);

    const stopAfterLabel = document.createElement('label');
    stopAfterLabel.textContent = 'Stop after matches';
    labelStyle(stopAfterLabel);

    const stopAfterInput = document.createElement('input');
    stopAfterInput.type = 'number';
    stopAfterInput.min = '0';
    stopAfterInput.step = '1';
    stopAfterInput.placeholder = '0';
    stopAfterInput.value = String(SETTINGS.stopAfterNMatches);
    stopAfterInput.id = 'tm-reddit-scraper-stop-after';
    inputStyle(stopAfterInput);
    stopAfterInput.addEventListener('input', () => {
      SETTINGS.stopAfterNMatches = Math.max(0, parseInt(stopAfterInput.value || '0', 10) || 0);
    });

    stopAfterWrap.appendChild(stopAfterLabel);
    stopAfterWrap.appendChild(stopAfterInput);

    controlsWrap.appendChild(timeframeWrap);
    controlsWrap.appendChild(minScoreWrap);
    controlsWrap.appendChild(maxScoreWrap);
    controlsWrap.appendChild(minCommentsWrap);
    controlsWrap.appendChild(stopAfterWrap);

    filtersSection.appendChild(filtersTitle);
    filtersSection.appendChild(filtersHelp);
    filtersSection.appendChild(controlsWrap);

    const actionsSection = document.createElement('div');
    sectionStyle(actionsSection);

    const actionsTitle = document.createElement('div');
    sectionTitleStyle(actionsTitle);
    actionsTitle.textContent = 'Export';

    const actionsHelp = document.createElement('p');
    helpTextStyle(actionsHelp);
    actionsHelp.textContent = 'Exports use the current filters and visible listing state.';

    const buttons = document.createElement('div');
    buttons.className = 'tm-rs-actions';

    const jsonBtn = document.createElement('button');
    jsonBtn.textContent = 'Export JSON';
    buttonStyle(jsonBtn, 'orange');
    jsonBtn.addEventListener('click', onClickWrapped(runScrapeAndDownloadJSON));

    const csvBtn = document.createElement('button');
    csvBtn.textContent = 'Export CSV';
    buttonStyle(csvBtn, 'blue');
    csvBtn.addEventListener('click', onClickWrapped(runScrapeAndDownloadCSV));

    const clipBtn = document.createElement('button');
    clipBtn.textContent = 'Copy JSON';
    buttonStyle(clipBtn, 'green');
    clipBtn.addEventListener('click', onClickWrapped(runScrapeToClipboard));

    const fullBtn = document.createElement('button');
    fullBtn.textContent = 'Posts + Comments';
    buttonStyle(fullBtn, 'violet');
    fullBtn.addEventListener('click', onClickWrapped(runFullPostsAndCommentsExport));

    const toggleLogBtn = document.createElement('button');
    toolbarButtonStyle(toggleLogBtn);

    const hideBtn = document.createElement('button');
    toolbarButtonStyle(hideBtn);
    hideBtn.textContent = 'Hide';

    const logWrap = document.createElement('div');
    sectionStyle(logWrap);
    logWrap.style.display = 'none';

    const logTitle = document.createElement('div');
    sectionTitleStyle(logTitle);
    logTitle.textContent = 'Activity log';

    const logHelp = document.createElement('p');
    helpTextStyle(logHelp);
    logHelp.textContent = 'Detailed progress appears here when you need to inspect a run.';

    const updateLogToggleLabel = () => {
      toggleLogBtn.textContent = logWrap.style.display === 'none' ? 'Show Log' : 'Hide Log';
    };

    toggleLogBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      logWrap.style.display = logWrap.style.display === 'none' ? 'block' : 'none';
      updateLogToggleLabel();
    });

    hideBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      panel.style.display = 'none';
      showRestoreBubble();
    });
    updateLogToggleLabel();

    buttons.appendChild(jsonBtn);
    buttons.appendChild(csvBtn);
    buttons.appendChild(clipBtn);
    buttons.appendChild(fullBtn);

    actionsSection.appendChild(actionsTitle);
    actionsSection.appendChild(actionsHelp);
    actionsSection.appendChild(buttons);

    const log = document.createElement('pre');
    log.id = LOG_ID;
    log.className = 'tm-rs-log';

    logWrap.appendChild(logTitle);
    logWrap.appendChild(logHelp);
    logWrap.appendChild(log);

    headerActions.appendChild(toggleLogBtn);
    headerActions.appendChild(hideBtn);
    header.appendChild(titleWrap);
    header.appendChild(headerActions);

    panel.appendChild(header);
    panel.appendChild(statusWrap);
    panel.appendChild(filtersSection);
    panel.appendChild(actionsSection);
    panel.appendChild(logWrap);

    document.body.appendChild(panel);
    enablePanelDragging(panel, header);
  }

  function onClickWrapped(fn) {
    return function (e) {
      e.preventDefault();
      e.stopPropagation();
      fn();
    };
  }

  function showRestoreBubble() {
    if (document.getElementById(RESTORE_ID)) return;
    ensurePanelStyles();

    const bubble = document.createElement('button');
    bubble.id = RESTORE_ID;
    bubble.textContent = 'Open Scraper';
    bubble.type = 'button';

    bubble.addEventListener('click', () => {
      bubble.remove();
      const panel = document.getElementById(PANEL_ID);
      if (panel) panel.style.display = 'block';
    });

    document.body.appendChild(bubble);
    enableBubbleDragging(bubble);
  }

  function ensurePanelExists() {
    if (!document.body) return;
    createPanel();
  }

  function startPanelObserver() {
    if (observerStarted) return;
    observerStarted = true;

    const observer = new MutationObserver(() => {
      if (!document.getElementById(PANEL_ID) && !document.getElementById(RESTORE_ID)) {
        createPanel();
      }
    });

    observer.observe(document.documentElement || document.body, {
      childList: true,
      subtree: true
    });

    window.addEventListener('popstate', () => {
      setTimeout(ensurePanelExists, 300);
    });

    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) setTimeout(ensurePanelExists, 100);
    });
  }

  function readTextCandidates(root, selectors) {
    for (const selector of selectors) {
      const el = querySelectorSafe(root, selector);
      const text = cleanText(el?.textContent);
      if (text) return text;
    }
    return '';
  }

  function readAttrCandidates(root, attrs) {
    for (const attr of attrs) {
      const value = cleanText(root.getAttribute?.(attr));
      if (value) return value;
    }
    return '';
  }

  function pickBestText(...values) {
    return values.map(v => cleanText(v)).find(Boolean) || '';
  }

  function extractPostIdFromUrl(url) {
    const match = String(url || '').match(/\/comments\/([a-z0-9]+)\//i);
    return match ? match[1] : '';
  }

  function getPostKey(post) {
    return post?.post_id || post?.permalink || '';
  }

  function mergePostRecords(existing, incoming) {
    if (!existing) return { ...incoming };

    const merged = { ...existing };

    for (const [key, value] of Object.entries(incoming || {})) {
      const oldValue = merged[key];

      if (value == null || value === '') continue;

      if (typeof value === 'number') {
        if (typeof oldValue !== 'number' || value > oldValue) {
          merged[key] = value;
        }
        continue;
      }

      if (typeof value === 'string') {
        if (!oldValue || String(value).length > String(oldValue).length) {
          merged[key] = value;
        }
        continue;
      }

      if (Array.isArray(value)) {
        if (!Array.isArray(oldValue) || value.length > oldValue.length) {
          merged[key] = value;
        }
        continue;
      }

      if (typeof value === 'object') {
        merged[key] = { ...(oldValue || {}), ...value };
      }
    }

    if (typeof existing.rank === 'number' && typeof incoming.rank === 'number') {
      merged.rank = Math.min(existing.rank, incoming.rank);
    }

    return merged;
  }

  function getCandidatePostContainers() {
    if (isOldReddit()) {
      const oldNodes = [
        ...document.querySelectorAll('.thing.link'),
        ...document.querySelectorAll('.thing[data-fullname]')
      ];
      return uniqueBy(oldNodes, el => (
        el.getAttribute('data-fullname') ||
        el.getAttribute('id') ||
        el.querySelector('a.comments')?.getAttribute('href') ||
        el.outerHTML.slice(0, 180)
      ));
    }

    const selectors = [
      'shreddit-post',
      '[data-testid="post-container"]',
      'faceplate-tracker[noun="post"]',
      'article',
      '[id^="t3_"]'
    ];

    const nodes = [];
    for (const selector of selectors) {
      querySelectorAllSafe(document, selector).forEach(el => {
        const hasCommentsLink =
          !!el.querySelector?.('a[href*="/comments/"]') ||
          !!el.querySelector?.('a[data-click-id="comments"]');
        const hasPermalink =
          !!el.getAttribute?.('permalink') ||
          !!el.getAttribute?.('post-id') ||
          !!el.getAttribute?.('thingid');

        if (el.matches?.('shreddit-post') || hasCommentsLink || hasPermalink) {
          nodes.push(el);
        }
      });
    }

    return uniqueBy(nodes, el => (
      el.getAttribute('id') ||
      el.getAttribute('thingid') ||
      el.getAttribute('post-id') ||
      el.getAttribute('data-post-id') ||
      el.getAttribute('permalink') ||
      el.querySelector('a[href*="/comments/"]')?.getAttribute('href') ||
      el.outerHTML.slice(0, 180)
    ));
  }

  function getMultilineTextLines(root) {
    const raw = String(root?.innerText || root?.textContent || '')
      .replace(/\u00A0/g, ' ')
      .replace(/\r/g, '');

    return raw
      .split(/\n+/)
      .map(line => cleanText(line))
      .filter(Boolean);
  }

  function findModernCommentsAnchor(root) {
    const selectors = [
      'a[href*="/comments/"]',
      'a[data-click-id="comments"]',
      'a[slot="full-post-link"]',
      'faceplate-tracker a[href*="/comments/"]'
    ];

    for (const selector of selectors) {
      const anchor = querySelectorSafe(root, selector);
      if (anchor) return anchor;
    }

    return null;
  }

  function findModernScoreText(root) {
    const attrCandidates = [
      root.getAttribute?.('score'),
      root.getAttribute?.('vote-count'),
      root.getAttribute?.('upvote-count'),
      root.getAttribute?.('aria-label')
    ];

    for (const value of attrCandidates) {
      const parsed = parseScore(value);
      if (parsed != null) return String(value);
    }

    const nodeSelectors = [
      'faceplate-number[aria-label*="vote"]',
      'faceplate-number[aria-label*="point"]',
      '[data-testid="vote-arrows"] [aria-label*="vote"]',
      '[aria-label*="votes"]',
      '[aria-label*="points"]'
    ];

    for (const selector of nodeSelectors) {
      const node = querySelectorSafe(root, selector);
      if (!node) continue;
      const candidates = [
        node.getAttribute?.('number'),
        node.getAttribute?.('aria-label'),
        node.textContent
      ];
      for (const candidate of candidates) {
        const parsed = parseScore(candidate);
        if (parsed != null) return String(candidate);
      }
    }

    const lines = getMultilineTextLines(root);
    for (const line of lines) {
      const parsed = parseScore(line);
      if (parsed != null) return line;
    }

    return '';
  }

  function findModernCommentsText(root) {
    const attrCandidates = [
      root.getAttribute?.('comment-count'),
      root.getAttribute?.('num-comments'),
      root.getAttribute?.('comments')
    ];

    for (const value of attrCandidates) {
      const parsed = parseComments(value);
      if (parsed != null) return String(value);
    }

    const nodeSelectors = [
      '[aria-label*="comment"]',
      'faceplate-number[aria-label*="comment"]',
      'a[data-click-id="comments"]'
    ];

    for (const selector of nodeSelectors) {
      const nodes = querySelectorAllSafe(root, selector);
      for (const node of nodes) {
        const candidates = [
          node.getAttribute?.('number'),
          node.getAttribute?.('aria-label'),
          node.textContent
        ];
        for (const candidate of candidates) {
          const parsed = parseComments(candidate);
          if (parsed != null) return String(candidate);
        }
      }
    }

    const lines = getMultilineTextLines(root);
    for (const line of lines) {
      const parsed = parseComments(line);
      if (parsed != null) return line;
    }

    return '';
  }

  function extractFromOldReddit(el, rankIndex, pageMeta) {
    const titleLink = el.querySelector('a.title, a.title.may-blank');
    const commentsLink = el.querySelector('a.comments');
    const scoreText = cleanText(el.querySelector('.score.unvoted, .score.likes, .score.dislikes')?.textContent);
    const commentText = cleanText(commentsLink?.textContent);
    const flair = cleanText(el.querySelector('.linkflairlabel')?.textContent);
    const author = cleanText(el.querySelector('.author')?.textContent);
    const subreddit = cleanText(el.querySelector('.subreddit')?.textContent) || (pageMeta.subreddit ? `r/${pageMeta.subreddit}` : '');
    const created = cleanText(el.querySelector('time')?.getAttribute('datetime')) || cleanText(el.querySelector('.tagline time')?.textContent);
    const permalink = normalizeCommentsPermalink(absoluteUrl(commentsLink?.getAttribute('href'), location.origin));
    const body = cleanText(el.querySelector('.expando .usertext-body')?.textContent);

    return {
      rank: rankIndex,
      post_id: extractPostIdFromUrl(permalink) || cleanText(el.getAttribute('data-fullname')).replace(/^t3_/, ''),
      title: cleanText(titleLink?.textContent),
      permalink,
      author,
      subreddit,
      subreddit_slug: pageMeta.subreddit,
      score: parseScore(scoreText),
      comments: parseComments(commentText),
      flair,
      created,
      body,
      sort: pageMeta.sort,
      timeframe: pageMeta.timeframe,
      page_url: pageMeta.url,
      scraped_at: new Date().toISOString()
    };
  }

  function extractFromShredditPost(el, rankIndex, pageMeta) {
    const commentsAnchor = findModernCommentsAnchor(el);
    const permalink = normalizeCommentsPermalink(
      absoluteUrl(
        el.getAttribute('permalink') ||
        commentsAnchor?.getAttribute('href') ||
        commentsAnchor?.getAttribute('permalink'),
        location.href
      )
    );

    const title = pickBestText(
      el.getAttribute('post-title'),
      readTextCandidates(el, [
        '[slot="title"]',
        '[slot="headline"]',
        'a[slot="title"]',
        '[data-testid="post-title"]',
        '[id^="post-title"]',
        'h1',
        'h2',
        'h3'
      ]),
      cleanText(commentsAnchor?.textContent)
    );

    const author = pickBestText(
      el.getAttribute('author'),
      readTextCandidates(el, [
        'a[href*="/user/"]',
        '[data-testid="post_author_link"]',
        '[data-testid="post-author-link"]'
      ])
    );

    const subreddit = pickBestText(
      el.getAttribute('subreddit-prefixed-name'),
      el.getAttribute('subreddit-name-prefixed'),
      readTextCandidates(el, [
        'a[href^="/r/"]',
        '[data-testid="subreddit-name"]'
      ]),
      pageMeta.subreddit ? `r/${pageMeta.subreddit}` : ''
    );

    const created = pickBestText(
      el.getAttribute('created-timestamp'),
      el.getAttribute('created'),
      cleanText(el.querySelector('time')?.getAttribute('datetime')),
      cleanText(el.querySelector('time')?.textContent)
    );

    const flair = pickBestText(
      el.getAttribute('post-flair-text'),
      readTextCandidates(el, [
        '[slot="post-flair"]',
        '[data-testid="post-flair"]',
        '[id*="flair"]'
      ])
    );

    const commentText = findModernCommentsText(el);
    const scoreText = findModernScoreText(el);

    const bodyText = pickBestText(
      el.getAttribute('content'),
      readTextCandidates(el, [
        '[slot="text-body"]',
        '[data-click-id="text"]',
        '[data-testid="post-content"]',
        '[data-post-click-location="text-body"]',
        '.md'
      ])
    );

    return {
      rank: rankIndex,
      post_id: extractPostIdFromUrl(permalink) || readAttrCandidates(el, ['post-id', 'id', 'thingid']),
      title,
      permalink,
      author,
      subreddit,
      subreddit_slug: pageMeta.subreddit,
      score: parseScore(scoreText),
      comments: parseComments(commentText),
      flair,
      created,
      body: bodyText,
      sort: pageMeta.sort,
      timeframe: pageMeta.timeframe,
      page_url: pageMeta.url,
      scraped_at: new Date().toISOString()
    };
  }

  function extractGeneric(el, rankIndex, pageMeta) {
    const commentAnchor = findModernCommentsAnchor(el);
    const permalink = normalizeCommentsPermalink(
      absoluteUrl(
        commentAnchor?.getAttribute('href') || commentAnchor?.getAttribute('permalink'),
        location.href
      )
    );

    const title = pickBestText(
      readTextCandidates(el, [
        '[data-testid="post-title"]',
        '[id^="post-title"]',
        '[slot="title"]',
        '[slot="headline"]',
        'h3',
        'h2',
        'h1',
        'a[slot="title"]'
      ]),
      cleanText(commentAnchor?.textContent)
    );

    const author = pickBestText(
      readTextCandidates(el, [
        'a[href*="/user/"]',
        'a[data-testid="post_author_link"]',
        'a[data-testid="post-author-link"]'
      ])
    );

    const subreddit = pickBestText(
      readTextCandidates(el, [
        'a[href^="/r/"]',
        '[data-testid="subreddit-name"]'
      ]),
      pageMeta.subreddit ? `r/${pageMeta.subreddit}` : ''
    );

    const created = pickBestText(
      cleanText(el.querySelector('time')?.getAttribute('datetime')),
      cleanText(el.querySelector('time')?.textContent)
    );

    const flair = pickBestText(
      readTextCandidates(el, [
        '[data-testid="post-flair"]',
        '[id*="flair"]',
        '[slot="post-flair"]'
      ])
    );

    const body = pickBestText(
      readTextCandidates(el, [
        '[data-click-id="text"]',
        '[data-testid="post-content"]',
        '[slot="text-body"]',
        '[data-post-click-location="text-body"]',
        '.md'
      ])
    );

    const textLines = getMultilineTextLines(el);

    let score = null;
    let comments = null;

    for (const line of textLines) {
      if (comments == null) {
        const maybeComments = parseComments(line);
        if (maybeComments != null) {
          comments = maybeComments;
          continue;
        }
      }

      if (score == null) {
        const maybeScore = parseScore(line);
        if (maybeScore != null) {
          score = maybeScore;
        }
      }

      if (score != null && comments != null) break;
    }

    if (score == null) {
      score = parseScore(findModernScoreText(el));
    }

    if (comments == null) {
      comments = parseComments(findModernCommentsText(el));
    }

    return {
      rank: rankIndex,
      post_id: extractPostIdFromUrl(permalink) || readAttrCandidates(el, ['post-id', 'id', 'thingid', 'data-post-id']),
      title,
      permalink,
      author,
      subreddit,
      subreddit_slug: pageMeta.subreddit,
      score,
      comments,
      flair,
      created,
      body,
      sort: pageMeta.sort,
      timeframe: pageMeta.timeframe,
      page_url: pageMeta.url,
      scraped_at: new Date().toISOString()
    };
  }

  function scrapeVisiblePosts() {
    const pageMeta = getUrlInfo();
    const containers = getCandidatePostContainers();
    const results = [];

    containers.forEach((el, index) => {
      let post;
      if (isOldReddit()) {
        post = extractFromOldReddit(el, index + 1, pageMeta);
      } else {
        const tagName = el.tagName.toLowerCase();
        post = tagName === 'shreddit-post'
          ? extractFromShredditPost(el, index + 1, pageMeta)
          : extractGeneric(el, index + 1, pageMeta);
      }

      if (post.title && post.permalink && /\/comments\//.test(post.permalink)) {
        results.push(post);
      }
    });

    return uniqueBy(results, post => getPostKey(post));
  }

  function getFilters() {
    return {
      timeframe: SETTINGS.timeframe,
      minScore: Math.max(0, SETTINGS.minScore || 0),
      maxScore: Math.max(0, SETTINGS.maxScore || 0),
      minComments: Math.max(0, SETTINGS.minComments || 0),
      stopAfterNMatches: Math.max(0, SETTINGS.stopAfterNMatches || 0)
    };
  }

  function passesFilters(post, filters) {
    const score = typeof post.score === 'number' ? post.score : 0;
    const comments = typeof post.comments === 'number' ? post.comments : 0;
    const passesMinScore = score >= filters.minScore;
    const passesMaxScore = filters.maxScore <= 0 || score <= filters.maxScore;
    const passesMinComments = comments >= filters.minComments;
    return passesMinScore && passesMaxScore && passesMinComments;
  }

  function buildTimeframeUrl() {
    const url = new URL(location.href);
    const info = getUrlInfo();
    const timeframe = SETTINGS.timeframe;

    if (info.subreddit) {
      if (!/\/top(\/|$)/.test(url.pathname)) {
        const parts = url.pathname.split('/').filter(Boolean);
        const rIndex = parts.findIndex(p => p.toLowerCase() === 'r');
        if (rIndex !== -1 && parts[rIndex + 1]) {
          url.pathname = `/r/${parts[rIndex + 1]}/top/`;
        }
      }
      url.searchParams.set('t', timeframe);
    }

    return url.toString();
  }

  async function ensureTargetTimeframePage() {
    const current = getUrlInfo();
    const targetTimeframe = SETTINGS.timeframe;

    if (!current.subreddit) {
      appendLog('No subreddit detected in URL. Using current page.');
      return;
    }

    const targetUrl = buildTimeframeUrl();

    if (targetUrl !== location.href) {
      setStatus(`Switching to timeframe: ${targetTimeframe}`);
      appendLog(`Navigating to ${targetUrl}`);
      location.href = targetUrl;
      return new Promise(() => {});
    }

    appendLog(`Using timeframe ${targetTimeframe} on current page.`);
  }

  async function autoScrollAndCollectFilteredPosts() {
    const filters = getFilters();
    let noGrowthSteps = 0;
    let previousVisibleCount = 0;
    let previousMatchCount = 0;

    const seenVisible = new Map();
    const seenMatching = new Map();

    function upsert(map, posts) {
      for (const post of posts) {
        const key = getPostKey(post);
        if (!key) continue;
        map.set(key, mergePostRecords(map.get(key), post));
      }
    }

    function getSortedMatchingPosts() {
      const posts = [...seenMatching.values()]
        .sort((a, b) => (a.rank || Number.MAX_SAFE_INTEGER) - (b.rank || Number.MAX_SAFE_INTEGER));
      if (filters.stopAfterNMatches > 0) {
        return posts.slice(0, filters.stopAfterNMatches);
      }
      return posts;
    }

    for (let step = 0; step < CONFIG.autoScroll.maxSteps; step++) {
      const visiblePosts = scrapeVisiblePosts();
      upsert(seenVisible, visiblePosts);
      upsert(seenMatching, visiblePosts.filter(post => passesFilters(post, filters)));

      const matchingNow = getSortedMatchingPosts();

      setStatus(`Auto-scrolling... step ${step + 1}, visible seen: ${seenVisible.size}, matching seen: ${matchingNow.length}`);
      appendLog(`Auto-scroll step ${step + 1}, visible seen: ${seenVisible.size}, matching seen: ${matchingNow.length}`);

      if (filters.stopAfterNMatches > 0 && matchingNow.length >= filters.stopAfterNMatches) {
        appendLog(`Reached stop-after limit of ${filters.stopAfterNMatches} matching posts.`);
        break;
      }

      window.scrollTo(0, document.body.scrollHeight);
      await sleep(CONFIG.autoScroll.delayMs);
      await sleep(CONFIG.autoScroll.settleMs);

      const nextVisiblePosts = scrapeVisiblePosts();
      upsert(seenVisible, nextVisiblePosts);
      upsert(seenMatching, nextVisiblePosts.filter(post => passesFilters(post, filters)));

      const visibleCount = seenVisible.size;
      const matchCount = seenMatching.size;

      if (visibleCount <= previousVisibleCount && matchCount <= previousMatchCount) {
        noGrowthSteps += 1;
      } else {
        noGrowthSteps = 0;
      }

      previousVisibleCount = visibleCount;
      previousMatchCount = matchCount;

      if (noGrowthSteps >= CONFIG.autoScroll.stopAfterNoGrowthSteps) {
        appendLog(`Stopping auto-scroll after ${step + 1} steps. No more growth.`);
        break;
      }
    }

    window.scrollTo(0, 0);
    await sleep(250);

    const finalPosts = getSortedMatchingPosts();
    appendLog(`Collected ${finalPosts.length} matching posts after filters across all scroll steps.`);
    return finalPosts;
  }

  function download(filename, text, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function toCSV(rows) {
    if (!rows.length) return '';

    const columns = [
      'rank',
      'post_id',
      'title',
      'permalink',
      'author',
      'subreddit',
      'subreddit_slug',
      'score',
      'comments',
      'flair',
      'created',
      'body',
      'sort',
      'timeframe',
      'page_url',
      'scraped_at'
    ];

    const escapeCSV = value => `"${String(value ?? '').replace(/"/g, '""')}"`;

    return [
      columns.join(','),
      ...rows.map(row => columns.map(col => escapeCSV(row[col])).join(','))
    ].join('\n');
  }

  function buildFilename(ext, suffix = 'posts') {
    const meta = getUrlInfo();
    const filters = getFilters();
    const subreddit = safeFilenamePart(meta.subreddit || 'reddit');
    const sort = safeFilenamePart(meta.sort || 'listing');
    const timeframe = safeFilenamePart(filters.timeframe || meta.timeframe || 'default');
    const host = safeFilenamePart(meta.is_old_reddit ? 'oldreddit' : 'reddit');
    const minScore = safeFilenamePart(`minscore${filters.minScore}`);
    const maxScore = safeFilenamePart(`maxscore${filters.maxScore || 'all'}`);
    const comments = safeFilenamePart(`mincomments${filters.minComments}`);
    const stopAfter = safeFilenamePart(`limit${filters.stopAfterNMatches || 'all'}`);
    return `${host}_${subreddit}_${sort}_${timeframe}_${minScore}_${maxScore}_${comments}_${stopAfter}_${suffix}.${ext}`;
  }

  async function withRunLock(taskName, fn) {
    if (isRunning) {
      setStatus(`Already running: ${taskName}`);
      appendLog(`Blocked duplicate run: ${taskName}`);
      return;
    }

    isRunning = true;
    clearLog();
    appendLog(`${taskName} started.`);
    try {
      setStatus(`${taskName}: starting...`);
      await fn();
      appendLog(`${taskName} completed.`);
    } catch (err) {
      console.error('[Reddit Scraper] Error:', err);
      setStatus(`${taskName}: failed`);
      appendLog(`${taskName} failed: ${err?.message || String(err)}`);
      alert(`Reddit scraper failed: ${err?.message || err}`);
    } finally {
      isRunning = false;
    }
  }

  async function fetchHtmlDocument(url) {
    const res = await fetch(url, {
      credentials: 'include',
      headers: {
        'accept': 'text/html,application/xhtml+xml'
      }
    });

    if (!res.ok) {
      throw new Error(`Fetch failed ${res.status} for ${url}`);
    }

    const html = await res.text();
    const parser = new DOMParser();
    return parser.parseFromString(html, 'text/html');
  }

  function parseCommentDepth(el) {
    const attrCandidates = ['depth', 'comment-depth', 'nesting-level'];
    const attrValue = readAttrCandidates(el, attrCandidates);
    if (attrValue && /^\d+$/.test(attrValue)) return Number(attrValue);

    const ariaLevel = el.getAttribute?.('aria-level');
    if (ariaLevel && /^\d+$/.test(ariaLevel)) return Number(ariaLevel) - 1;

    let depth = 0;
    let node = el.parentElement;
    while (node) {
      const tag = node.tagName ? node.tagName.toLowerCase() : '';
      if (
        tag === 'shreddit-comment' ||
        node.matches?.('[data-testid="comment"]') ||
        node.classList?.contains('thing')
      ) {
        depth += 1;
      }
      node = node.parentElement;
    }
    return Math.max(0, depth - 1);
  }

  function findPostTitleInDoc(doc, fallback) {
    if (doc.location?.hostname === 'old.reddit.com') {
      return (
        cleanText(doc.querySelector('a.title')?.textContent) ||
        cleanText(doc.querySelector('.entry .title')?.textContent) ||
        cleanText(doc.querySelector('title')?.textContent.replace(/\s*:\s*reddit.*$/i, '')) ||
        fallback ||
        ''
      );
    }

    return (
      cleanText(doc.querySelector('shreddit-post')?.getAttribute('post-title')) ||
      cleanText(doc.querySelector('[data-testid="post-title"]')?.textContent) ||
      cleanText(doc.querySelector('[id^="post-title"]')?.textContent) ||
      cleanText(doc.querySelector('h1')?.textContent) ||
      fallback ||
      ''
    );
  }

  function extractPostBodyFromDocument(doc) {
    if (doc.location?.hostname === 'old.reddit.com') {
      const oldSelectors = [
        '.entry .usertext-body',
        '.top-matter .usertext-body',
        '.expando .usertext-body',
        '.selftext'
      ];
      for (const selector of oldSelectors) {
        const text = cleanText(doc.querySelector(selector)?.textContent);
        if (text) return text;
      }
      return '';
    }

    const shredditPost = doc.querySelector('shreddit-post');
    if (shredditPost) {
      const attrContent = cleanText(shredditPost.getAttribute('content'));
      if (attrContent) return attrContent;
      const slotBody = cleanText(shredditPost.querySelector('[slot="text-body"]')?.textContent);
      if (slotBody) return slotBody;
    }

    const selectors = [
      '[data-click-id="text"]',
      '[data-testid="post-content"]',
      '[slot="text-body"]',
      '[data-post-click-location="text-body"]',
      'div[data-adclicklocation="title"] ~ div',
      '.md'
    ];

    for (const selector of selectors) {
      const els = querySelectorAllSafe(doc, selector);
      for (const el of els) {
        const text = cleanText(el.textContent);
        if (text && text.length > 20) return text;
      }
    }

    return '';
  }

  function extractCommentsFromOldRedditDocument(doc) {
    const comments = [...doc.querySelectorAll('.comment.thing')];
    const results = [];

    for (const el of comments) {
      const body = cleanText(el.querySelector('.usertext-body')?.textContent);
      if (!body) continue;

      const author = cleanText(el.querySelector('.author')?.textContent);
      const created = cleanText(el.querySelector('time')?.getAttribute('datetime')) || cleanText(el.querySelector('time')?.textContent);
      const scoreText = cleanText(el.querySelector('.score.unvoted, .score.likes, .score.dislikes')?.textContent);
      const permalink = normalizeCommentsPermalink(absoluteUrl(el.querySelector('a.bylink')?.getAttribute('href'), doc.location?.origin || location.origin));
      const commentId = cleanText(el.getAttribute('data-fullname')).replace(/^t1_/, '');
      const parentId = cleanText(el.getAttribute('data-parent-fullname')).replace(/^t1_/, '').replace(/^t3_/, '');
      const depth = parseCommentDepth(el);

      results.push({
        comment_id: commentId,
        parent_id: parentId,
        depth,
        author,
        score: parseScore(scoreText),
        created,
        permalink,
        body
      });
    }

    return uniqueBy(results, c => c.comment_id || `${c.author}|${c.created}|${c.body.slice(0, 120)}`);
  }

  function extractCommentsFromDocument(doc) {
    if (doc.location?.hostname === 'old.reddit.com') {
      return extractCommentsFromOldRedditDocument(doc);
    }

    const candidates = uniqueBy(
      [
        ...querySelectorAllSafe(doc, 'shreddit-comment'),
        ...querySelectorAllSafe(doc, '[data-testid="comment"]'),
        ...querySelectorAllSafe(doc, 'article[data-testid="comment"]'),
        ...querySelectorAllSafe(doc, 'faceplate-comment'),
        ...querySelectorAllSafe(doc, '[thingid^="t1_"]')
      ],
      el => (
        readAttrCandidates(el, ['thingid', 'comment-id', 'id']) ||
        el.querySelector('a[href*="/comments/"]')?.getAttribute('href') ||
        cleanText(el.textContent).slice(0, 120)
      )
    );

    const results = [];

    for (const el of candidates) {
      const body = pickBestText(
        el.getAttribute?.('comment-body'),
        readTextCandidates(el, [
          '[slot="comment"]',
          '[data-testid="comment-content"]',
          '[data-click-id="text"]',
          '.md'
        ])
      );

      if (!body || body.length < 2) continue;

      const author = pickBestText(
        el.getAttribute?.('author'),
        readTextCandidates(el, [
          'a[href*="/user/"]',
          '[data-testid="comment_author_link"]',
          '[data-testid="comment-author-link"]'
        ])
      );

      const created = pickBestText(
        el.getAttribute?.('created-timestamp'),
        cleanText(el.querySelector?.('time')?.getAttribute('datetime')),
        cleanText(el.querySelector?.('time')?.textContent)
      );

      const scoreText = pickBestText(
        el.getAttribute?.('score'),
        el.getAttribute?.('vote-count'),
        cleanText(el.querySelector?.('faceplate-number[aria-label*="point"], faceplate-number[aria-label*="vote"]')?.getAttribute('number')),
        cleanText(el.querySelector?.('[aria-label*="point"], [aria-label*="vote"]')?.getAttribute('aria-label'))
      );

      const permalink = normalizeCommentsPermalink(
        absoluteUrl(el.querySelector?.('a[href*="/comments/"]')?.getAttribute('href'), location.href)
      );

      const commentId =
        readAttrCandidates(el, ['comment-id', 'id', 'thingid']) ||
        (permalink.match(/\/([a-z0-9]+)$/i)?.[1] || '');

      const parentId =
        readAttrCandidates(el, ['parent-id', 'parent-fullname']) ||
        '';

      const depth = parseCommentDepth(el);

      results.push({
        comment_id: cleanText(commentId).replace(/^t1_/, ''),
        parent_id: cleanText(parentId).replace(/^t1_/, '').replace(/^t3_/, ''),
        depth,
        author,
        score: parseScore(scoreText),
        created,
        permalink,
        body
      });
    }

    return uniqueBy(results, c => c.comment_id || `${c.author}|${c.created}|${c.body.slice(0, 120)}`);
  }

  async function scrapeFullPostPage(post) {
    let lastErr = null;

    for (let attempt = 0; attempt <= CONFIG.fullScrape.retryCount; attempt++) {
      try {
        const doc = await fetchHtmlDocument(post.permalink);
        const title = findPostTitleInDoc(doc, post.title);
        const bodyFull = extractPostBodyFromDocument(doc);
        const commentsFull = extractCommentsFromDocument(doc);

        return {
          ...post,
          title: title || post.title,
          body_full: bodyFull || post.body || '',
          comments_full_count: commentsFull.length,
          comments_full: commentsFull
        };
      } catch (err) {
        lastErr = err;
        if (attempt < CONFIG.fullScrape.retryCount) {
          appendLog(`Retry ${attempt + 1} for ${post.permalink}`);
          await sleep(CONFIG.fullScrape.retryDelayMs);
        }
      }
    }

    throw lastErr;
  }

  async function collectMatchingPosts() {
    await ensureTargetTimeframePage();
    return await autoScrollAndCollectFilteredPosts();
  }

  async function runScrapeAndDownloadJSON() {
    return withRunLock('JSON export', async () => {
      const posts = await collectMatchingPosts();
      const filters = getFilters();

      const payload = {
        meta: {
          ...getUrlInfo(),
          scraped_at: new Date().toISOString(),
          count: posts.length,
          mode: 'listing_only',
          filters
        },
        posts
      };

      download(
        buildFilename('json', 'posts'),
        JSON.stringify(payload, null, 2),
        'application/json'
      );

      appendLog(`Downloaded JSON for ${posts.length} matching posts.`);
      setStatus(`Downloaded JSON: ${posts.length} matching posts`);
      alert(`Scraped ${posts.length} matching posts and downloaded JSON.`);
    });
  }

  async function runScrapeAndDownloadCSV() {
    return withRunLock('CSV export', async () => {
      const posts = await collectMatchingPosts();

      download(
        buildFilename('csv', 'posts'),
        toCSV(posts),
        'text/csv;charset=utf-8'
      );

      appendLog(`Downloaded CSV for ${posts.length} matching posts.`);
      setStatus(`Downloaded CSV: ${posts.length} matching posts`);
      alert(`Scraped ${posts.length} matching posts and downloaded CSV.`);
    });
  }

  async function runScrapeToClipboard() {
    return withRunLock('Clipboard export', async () => {
      const posts = await collectMatchingPosts();
      GM_setClipboard(JSON.stringify(posts, null, 2), 'text');

      appendLog(`Copied JSON for ${posts.length} matching posts to clipboard.`);
      setStatus(`Copied JSON: ${posts.length} matching posts`);
      alert(`Scraped ${posts.length} matching posts and copied JSON to clipboard.`);
    });
  }

  async function runFullPostsAndCommentsExport() {
    return withRunLock('Posts + comments export', async () => {
      const listingPosts = (await collectMatchingPosts()).slice(0, CONFIG.fullScrape.maxPosts);
      appendLog(`Found ${listingPosts.length} matching posts to fetch in full.`);

      const enrichedPosts = [];

      for (let i = 0; i < listingPosts.length; i++) {
        const post = listingPosts[i];
        const label = `${i + 1}/${listingPosts.length}`;
        setStatus(`Fetching ${label}: ${post.title.slice(0, 60)}`);
        appendLog(`Fetching ${label}: ${post.permalink}`);

        try {
          const enriched = await scrapeFullPostPage(post);
          enrichedPosts.push(enriched);
          appendLog(`Fetched ${label}: ${enriched.comments_full_count} comments`);
        } catch (err) {
          enrichedPosts.push({
            ...post,
            body_full: post.body || '',
            comments_full_count: 0,
            comments_full: [],
            fetch_error: String(err?.message || err)
          });
          appendLog(`Failed ${label}: ${err?.message || err}`);
        }

        if (i < listingPosts.length - 1) {
          await sleep(CONFIG.fullScrape.delayBetweenPostsMs);
        }
      }

      const totalComments = enrichedPosts.reduce((sum, p) => sum + (p.comments_full_count || 0), 0);
      const filters = getFilters();

      const payload = {
        meta: {
          ...getUrlInfo(),
          scraped_at: new Date().toISOString(),
          posts_count: enrichedPosts.length,
          comments_count: totalComments,
          mode: 'posts_with_full_content_and_comments',
          filters
        },
        posts: enrichedPosts
      };

      download(
        buildFilename('json', 'posts_comments'),
        JSON.stringify(payload, null, 2),
        'application/json'
      );

      appendLog(`Downloaded posts + comments JSON: ${enrichedPosts.length} posts, ${totalComments} comments.`);
      setStatus(`Downloaded posts+comments JSON: ${enrichedPosts.length} posts, ${totalComments} comments`);
      alert(`Scraped ${enrichedPosts.length} matching posts and ${totalComments} comments.`);
    });
  }

  GM_registerMenuCommand('Scrape Reddit posts -> Download JSON', runScrapeAndDownloadJSON);
  GM_registerMenuCommand('Scrape Reddit posts -> Download CSV', runScrapeAndDownloadCSV);
  GM_registerMenuCommand('Scrape Reddit posts -> Copy JSON to clipboard', runScrapeToClipboard);
  GM_registerMenuCommand('Scrape Reddit posts -> Download Posts + Comments JSON', runFullPostsAndCommentsExport);

  window.redditTopScraper = {
    scrapeVisiblePosts,
    runScrapeAndDownloadJSON,
    runScrapeAndDownloadCSV,
    runScrapeToClipboard,
    runFullPostsAndCommentsExport
  };

  function init() {
    ensurePanelExists();
    startPanelObserver();
    setStatus('Ready');
    appendLog(`Loaded on ${isOldReddit() ? 'old.reddit.com' : 'www.reddit.com'}. Timeframe default: ${SETTINGS.timeframe}.`);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
