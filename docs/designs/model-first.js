(() => {
  'use strict';
  const platform = document.body.dataset.platform;
  const app = document.getElementById('app');
  const dialog = document.getElementById('model-dialog');
  const names = { extract: 'スペックメモ', lab: '抽出ラボ', images: '画像検索', finder: 'Object finder', chat: 'Chat', work: 'Work', decide: 'Decide', speech: '音声合成' };
  const demoDefinitions = {
    extract: { title: '比較デモ', icon: 'compare', tone: 'green' },
    lab: { title: '抽出デモ', icon: 'gliner', tone: 'orange' },
    images: { title: '画像検索デモ', icon: 'embedding', tone: 'blue' },
    finder: { title: 'カメラデモ', icon: 'camera', tone: 'purple' }
  };
  const scaleDemos = Array.from({ length: 8 }, (_, index) => `layout-${index + 1}`);
  function demoDefinition(id) {
    return demoDefinitions[id] ?? { title: `配置例${id.split('-')[1]}デモ`, icon: ['gliner','compare','embedding','camera'][Number(id.split('-')[1]) % 4], tone: ['green','orange','blue','purple'][Number(id.split('-')[1]) % 4], layoutOnly: true };
  }
  const models = [
    { id: 'gliner', name: 'GLiNER2.5-multi', type: '情報抽出', category: 'Information Extraction', size: '1.2 GB', precision: '変換済みバンドル', description: '商品説明から、必要なスペックを取り出す。', demos: ['extract', 'lab'], ios: true },
    { id: 'embedding', name: 'EmbeddingGemma 2', type: '画像・テキスト', category: 'Multimodal Embedding', size: '0.9 GB', precision: 'FP16', description: '探したいものを言葉にして、画像を見つける。', demos: ['images', 'finder'], ios: true },
    { id: 'qwen', name: 'Qwen3.8 27B', type: '言語生成', category: 'LLM', size: '16.85 GB', precision: '4bit', description: '会話から、Webを使うタスクまで。', demos: ['chat', 'work'], ios: false, reason: 'macOS向けのモデルです。' },
    { id: 'clef', name: 'Clef-Flash 9B', type: '文書の判定', category: 'Document Scoring', size: '5.4 GB', precision: '4bit + FP16 head', description: '文書を読み、チェック項目ごとに判定する。', demos: ['decide'], ios: false, reason: 'このiOSアプリでは提供していません。' },
    { id: 'irodori', name: 'Irodori', type: '音声合成', category: 'TTS', size: '2.1 GB', precision: 'FP32', description: '入力したテキストを、自然な音声に。', demos: ['speech'], ios: false, reason: 'iOSの音声デモは別アプリで提供する案です。' }
  ];
  const iconPaths = {
    compare: '<rect x="3" y="5" width="7" height="15" rx="2"/><rect x="14" y="3" width="7" height="17" rx="2"/><path d="M5 10h3m-3 4h3m8-6h3m-3 4h3m-3 4h3"/>',
    trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14M10 11v6m4-6v6"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
    retry: '<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>',
    gliner: '<rect x="4" y="3" width="16" height="18" rx="3"/><path d="M8 8h8M8 12h5M8 16h7"/>',
    embedding: '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="8" cy="9" r="1.2"/><path d="m4 17 5-5 4 4 3-3 4 4"/>',
    qwen: '<path d="M20 11.5a8 8 0 0 1-8 8H5l-3 2 1.5-5A8.5 8.5 0 1 1 20 11.5Z"/><path d="M7 10h8M7 14h5"/>',
    clef: '<rect x="4" y="3" width="16" height="18" rx="3"/><path d="m8 8 1 1 2-2m-3 7 1 1 2-2m3-5h2m-2 6h2"/>',
    irodori: '<path d="M3 10v4m4-8v12m5-15v18m5-15v12m4-8v4"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.2"/>',
    back: '<path d="m14 5-7 7 7 7"/>',
    arrow: '<path d="M12 20V4m-6 6 6-6 6 6"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v9H3Z"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    download: '<path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/>',
    library: '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
    camera: '<path d="M3 8h4l2-3h6l2 3h4v12H3Z"/><circle cx="12" cy="14" r="4"/>'
  };
  const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${iconPaths[name] || iconPaths.library}</svg>`;
  const glyph = m => `<span class="model-glyph ${m.id}">${icon(m.id)}</span>`;
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state = { selected: null, demo: null, downloads: new Map(), query: '', installed: new Set(['gliner','qwen']), preparing: false, timer: null, values: {}, results: {}, explanations: {}, scale: false, searchActive: false, searchOriginOffset: 0, libraryOffset: 0, largeText: false };
  const getModel = id => models.find(m => m.id === id);
  const current = () => getModel(state.selected);
  const key = () => `${state.selected}:${state.demo}`;
  const available = m => platform === 'macos' || m.ios;
  const modelClassification = m => [m.category, m.architecture].filter(Boolean).join(' · ');
  const modelDemos = m => [...m.demos.filter(id => !(platform === 'macos' && id === 'finder')), ...(platform === 'ios' && state.scale && m.id === 'gliner' ? scaleDemos : [])];
  const canExecute = () => !!current() && available(current()) && modelDemos(current()).includes(state.demo) && state.installed.has(state.selected) && !state.demo.startsWith('layout-');
  function announce(text) { document.getElementById('announcement').textContent = text; }
  function focusHeading() { const target = app.querySelector('.demo-title, .screen-title, .library-empty h2'); if (target) { target.tabIndex = -1; target.focus({ preventScroll: true }); } }
  function matchingDemos(m) {
    const query = state.query.trim().toLowerCase();
    if (`${m.name} ${m.type} ${modelClassification(m)} ${m.description}`.toLowerCase().includes(query)) return modelDemos(m);
    return modelDemos(m).filter(id => `${demoDefinition(id).title} ${demoExplanations[id]?.summary ?? ''}`.toLowerCase().includes(query));
  }
  function filteredModels() {
    const source = platform === 'macos' ? ['qwen','gliner','embedding','clef','irodori'].map(getModel) : models;
    return source.filter(m => available(m) && (platform === 'ios' ? matchingDemos(m).length > 0 : `${m.name} ${m.type} ${modelClassification(m)} ${m.description}`.toLowerCase().includes(state.query.toLowerCase())));
  }
  function search() {
    const label = platform === 'ios' ? 'モデル・デモを検索' : 'モデルを検索';
    return `<div class="search">${icon('search')}<label class="sr-only" for="model-search">${label}</label><input id="model-search" type="search" data-search placeholder="${platform === 'ios' ? label : 'モデル名・用途で検索'}" value="${escape(state.query)}" autocomplete="off" enterkeyhint="search"><button type="button" class="search-clear" data-action="clear-search-field" aria-label="検索語を消去" ${state.query ? '' : 'hidden'}>${icon('close')}</button></div>`;
  }
  function searchToolbar() {
    return `<div class="search-toolbar ${state.searchActive ? 'is-searching' : ''}" role="search" aria-label="モデルとデモの検索">${search()}<button type="button" class="search-cancel" data-action="cancel-search" ${state.searchActive ? '' : 'hidden'}>キャンセル</button></div>`;
  }
  function activateSearch() {
    if (!state.searchActive) state.searchOriginOffset = app.querySelector('.phone-scroll')?.scrollTop ?? 0;
    state.searchActive = true;
    app.querySelector('.search-toolbar')?.classList.add('is-searching');
    const cancel = app.querySelector('.search-cancel');
    if (cancel) cancel.hidden = false;
  }
  function refreshSearch() {
    const list = document.getElementById('model-list');
    if (list) list.innerHTML = listContent();
    const count = document.getElementById('model-count');
    if (count) count.textContent = `${filteredModels().length}件`;
    const clear = app.querySelector('.search-clear');
    if (clear) clear.hidden = !state.query;
  }
  function clearSearch() {
    state.query = '';
    const input = app.querySelector('[data-search]');
    if (input) input.value = '';
    refreshSearch();
    input?.focus({preventScroll:true});
  }
  function cancelSearch() {
    state.query = '';
    state.searchActive = false;
    state.libraryOffset = state.searchOriginOffset;
    render();
    app.querySelector('.screen-title')?.focus({preventScroll:true});
    announce('検索を終了しました。');
  }
  function listContent() {
    const list = filteredModels();
    if (!list.length) return `<div class="empty-results">${icon('search')}<strong>検索結果がありません</strong><p>「${escape(state.query)}」に一致するモデル・デモがありません。</p><button type="button" data-action="clear-search-field">検索語を消去</button></div>`;
    return list.map(m => platform === 'macos' ? `<button type="button" class="sidebar-model" data-action="select" data-model="${m.id}" aria-pressed="${state.selected === m.id}">${glyph(m)}<span class="row-copy"><span class="model-name">${m.name}</span><span class="model-sub">${m.type} · ${state.installed.has(m.id) ? '取得済み' : '未取得'}</span></span><span class="side-size">${m.size}</span></button>` : modelSection(m)).join('');
  }
  function acquisitionIcons(m) {
    const download = state.downloads.get(m.id);
    if (state.installed.has(m.id)) return `<span class="model-installed" role="img" aria-label="${m.name}はダウンロード済み" title="ダウンロード済み">${icon('check')}</span><button type="button" class="model-action remove-model" data-action="request-delete" data-model="${m.id}" aria-label="${m.name}を削除" title="モデルを削除">${icon('trash')}</button>`;
    if (download?.phase === 'downloading') return `<span class="download-indicator"><span class="sr-only" data-ring-progress role="progressbar" aria-label="${m.name}のダウンロード" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${download.progress}"></span><button type="button" class="model-action download-ring" data-action="cancel-download" data-model="${m.id}" aria-label="${m.name}のダウンロードをキャンセル" title="ダウンロードをキャンセル"><svg viewBox="0 0 40 40" aria-hidden="true"><circle class="ring-track" cx="20" cy="20" r="16"/><circle class="ring-value" cx="20" cy="20" r="16" pathLength="100" stroke-dasharray="100" stroke-dashoffset="${100-download.progress}"/><path d="m16 16 8 8m-8 0 8-8"/></svg></button></span>`;
    const failed = download?.phase === 'failed';
    return `<button type="button" class="model-action ${failed ? 'download-failed' : ''}" data-action="download" data-model="${m.id}" aria-label="${m.name}${failed ? 'のダウンロードに失敗。再試行' : 'をダウンロード、' + m.size}" title="${failed ? 'ダウンロードに失敗。再試行' : 'ダウンロード · ' + m.size}">${icon(failed ? 'retry' : 'download')}</button>`;
  }
  function demoTile(m, id) {
    const definition = demoDefinition(id);
    return `<li><button type="button" class="demo-tile" data-action="open-demo" data-model="${m.id}" data-demo="${id}" aria-label="${m.name}の${definition.title}を開く${state.installed.has(m.id) ? '' : '。未ダウンロード、解説を閲覧できます'}"><span class="tile-art ${definition.tone}">${icon(definition.icon)}</span><span class="tile-title">${definition.title}</span></button></li>`;
  }
  function modelSection(m) {
    return `<section class="model-section" data-model-section="${m.id}" aria-labelledby="model-heading-${m.id}"><div class="model-section-heading"><div><h3 id="model-heading-${m.id}">${m.name}</h3><p>${modelClassification(m)} · ${m.size}</p></div><div class="model-actions" data-model-actions="${m.id}">${acquisitionIcons(m)}</div></div><ul class="demo-grid">${matchingDemos(m).map(id => demoTile(m,id)).join('')}</ul></section>`;
  }
  function library() {
    return `<div class="app-toolbar library-toolbar"><span class="nav-title" aria-hidden="true">Models</span><button type="button" class="icon-button" data-action="about" aria-label="このアプリについて">${icon('info')}</button></div><div class="phone-scroll"><h2 class="screen-title" tabindex="-1">Models</h2><div class="library-intro"><p class="screen-lead">モデルから、試したいデモを選ぶ。</p><p class="library-note">ダウンロード前でも、デモの解説を読めます。</p></div>${state.scale ? '<p class="scale-preview-label">10件の配置確認 · 追加8件は仮の項目です</p>' : ''}<div class="model-list launcher-list" id="model-list">${listContent()}</div></div>${searchToolbar()}`;
  }
  function refreshAcquisition(id) {
    for (const tile of app.querySelectorAll(`[data-action="open-demo"][data-model="${id}"]`)) {
      tile.setAttribute('aria-label', `${getModel(id).name}の${demoDefinition(tile.dataset.demo).title}を開く${state.installed.has(id) ? '' : '。未ダウンロード、解説を閲覧できます'}`);
    }
    const actions = app.querySelector(`[data-model-actions="${id}"]`);
    if (actions) {
      const restoreFocus = actions.contains(document.activeElement);
      actions.innerHTML = acquisitionIcons(getModel(id));
      if (restoreFocus) actions.querySelector('button')?.focus({preventScroll:true});
    }
    if (platform === 'ios' && state.selected === id) {
      const scroll = app.querySelector('.phone-scroll');
      const offset = scroll?.scrollTop ?? 0;
      render();
      app.querySelector('.phone-scroll').scrollTop = offset;
    }
  }
  function updateDownloadProgress(id, progress) {
    const ring = app.querySelector(`[data-model-actions="${id}"] .ring-value`);
    if (ring) {
      ring.setAttribute('stroke-dashoffset', 100-progress);
      app.querySelector(`[data-model-actions="${id}"] [data-ring-progress]`)?.setAttribute('aria-valuenow', progress);
    }
    if (state.selected === id) {
      const meter = app.querySelector('.access-progress');
      if (meter) meter.value = progress;
      const status = app.querySelector('[data-access-progress]');
      if (status) status.textContent = `ダウンロード中 ${progress}%`;
    }
  }
  function cancelDownloads() {
    for (const download of state.downloads.values()) clearInterval(download.timer);
    state.downloads.clear();
  }
  function startDownload(id) {
    const m = getModel(id);
    if (platform !== 'ios' || !m || !available(m) || state.installed.has(id) || state.downloads.get(id)?.phase === 'downloading') return;
    const download = { phase: 'downloading', progress: 0, timer: null };
    state.downloads.set(id, download);
    refreshAcquisition(id);
    announce(`${m.name}のダウンロード表示を開始しました。`);
    download.timer = setInterval(() => {
      if (state.downloads.get(id) !== download) return;
      download.progress = Math.min(100, download.progress + 10);
      if (download.progress === 100) {
        clearInterval(download.timer);
        state.downloads.delete(id);
        state.installed.add(id);
        refreshAcquisition(id);
        announce(`${m.name}のダウンロードが完了した表示です。デモを選ぶと実行できます。`);
        return;
      }
      updateDownloadProgress(id, download.progress);
    }, 600);
  }
  function cancelDownload(id) {
    const download = state.downloads.get(id);
    if (!download) return;
    clearInterval(download.timer);
    state.downloads.delete(id);
    refreshAcquisition(id);
    announce('ダウンロードをキャンセルしました。');
  }
  function previewDownload(failed = false) {
    cancelDownloads();
    state.selected = null;
    state.demo = null;
    state.query = '';
    state.searchActive = false;
    state.libraryOffset = 0;
    state.scale = false;
    state.installed.delete('embedding');
    if (failed) state.downloads.set('embedding', { phase: 'failed', progress: 0, timer: null });
    render(true);
  }
  function sidebar() {
    return `<aside class="model-sidebar" aria-label="モデル一覧"><div class="sidebar-heading"><h2>モデル</h2><button type="button" class="icon-button" data-action="about" aria-label="ライブラリについて">${icon('info')}</button></div>${search()}<p class="sidebar-label">このMacで使える · <span id="model-count">${filteredModels().length}件</span></p><div id="model-list">${listContent()}</div><p class="caption" style="padding:0 9px">容量はダウンロード量です。</p><button type="button" class="local-row" data-action="local">${icon('folder')}ローカルモデルを開く</button></aside>`;
  }
  function switcher() {
    const demos = modelDemos(current());
    return demos.length > 1 ? `<div class="demo-switch" role="group" aria-label="このモデルで試すデモ">${demos.map(id => `<button type="button" data-action="demo" data-demo="${id}" aria-pressed="${state.demo === id}">${names[id]}</button>`).join('')}</div>` : '';
  }
  function modelInfoButton() { return `<button type="button" class="icon-button" data-action="info" aria-label="${current().name}のモデル情報">${icon('info')}</button>`; }
  const demoExplanations = {
    extract: {
      summary: '商品説明から、容量や素材などのスペックを抜き出すデモです。',
      sections: [['試すこと', '選択したモデルを使い、文章の中から商品名・容量・素材・重量などの項目を取り出します。'], ['使い方', '商品説明を入力して「スペックを抽出」をタップします。「例を使う」からサンプル文でも試せます。'], ['結果の見方', '元の文章と抽出された項目を見比べ、抜けや取り違えがないかを確認します。文章にない情報を補う用途ではありません。']]
    },
    lab: {
      summary: '文章の表現を変えながら、指定した項目を取り出せるか試すデモです。',
      sections: [['試すこと', '同じモデルを使い、商品名・容量・素材・重量の抽出を試します。'], ['使い方', '入力例の言い回しや項目の順番を変えて「抽出する」をタップします。'], ['結果の見方', '入力に含めた項目が取り出せているか、別の項目と混ざっていないかを見比べます。']]
    },
    images: {
      summary: '探したいものを言葉で入力し、近い内容の画像を見つけるデモです。',
      sections: [['試すこと', '選択したモデルで画像と言葉を比べ、検索語に近い画像を探します。'], ['使い方', '「木の椅子」のように探したいものを入力し、画像の中から検索します。'], ['結果の見方', '言い回しを変えたときに、見つかる画像がどう変わるかを確認します。近さは、正しさを保証する確率ではありません。']]
    },
    finder: {
      summary: 'カメラの中から、指定した言葉に近いものを探すデモです。',
      sections: [['試すこと', '選択したモデルを使い、カメラに映る候補と言葉の近さを比べます。'], ['使い方', '「白いカップ」のように対象を入力し、カメラを向けて表示される枠を確認します。'], ['結果の見方', '対象や言葉を変えて、見つかるものと枠の変化を確認します。近い見た目のものを取り違える場合もあるため、結果を見比べて使います。']]
    }
  };
  function intro(title, description) {
    const m = current();
    if (platform === 'ios') {
      const explanation = demoExplanations[state.demo] ?? { summary: 'デモが10件になったときの配置と遷移を確認する仮の項目です。', sections: [['配置確認', '追加した8件はレイアウト確認用です。公開予定の機能や実装済みの機能を示すものではありません。']] };
      const disclosure = explanation ? `<details class="demo-explanation" data-explanation-key="${key()}" ${state.explanations[key()] ? 'open' : ''}><summary>このデモについて</summary><dl>${explanation.sections.map(([label, text]) => `<div><dt>${label}</dt><dd>${text}</dd></div>`).join('')}</dl></details>` : '';
      return `<h2 class="demo-title model-page-title">${m.name}</h2><p class="model-page-meta">${modelClassification(m)} · ${m.precision}</p><p class="current-demo-name">${demoDefinition(state.demo).title}</p><p class="demo-description">${explanation?.summary ?? description}</p>${disclosure}`;
    }
    return `<div class="demo-meta">${glyph(m)}<span>${m.name} · ${m.precision}</span></div><h2 class="demo-title">${title}</h2><p class="demo-description">${description}</p>`;
  }
  function prepare() {
    const m = current();
    return `${intro(names[state.demo], m.description)}<div class="prepare-panel">${glyph(m)}<h3>${state.preparing ? 'モデルを準備しています' : 'モデルを準備しましょう'}</h3><p>${state.preparing ? '準備が終わると、選んだデモが開きます。' : '最初の一度だけ、モデルの取得が必要です。'}</p><div class="prepare-facts"><span>モデル容量<strong>${m.size}</strong></span><span>使用するデモ<strong style="font-size:14px;margin-top:4px">${names[state.demo]}</strong></span></div>${state.preparing ? '<div class="progress-track" role="progressbar" aria-label="準備中"><span></span></div><p>準備中…</p><button type="button" class="secondary-button" data-action="cancel-prepare">キャンセル</button>' : `<button type="button" class="primary-button" data-action="prepare">${icon('download')}取得してデモを開く</button><button type="button" class="secondary-button" data-action="use-local">手元のモデルを開く</button>`}</div><p class="sample-message">このデザイン案では、準備完了までの画面遷移だけを再現します。</p>`;
  }
  const examples = {
    extract: 'Aero Bottle 500は、容量500mLのステンレス製ボトルです。重さは210g。保温は6時間、保冷は12時間。カラーはフォレストグリーン。',
    lab: 'Aero Bottle 500は、容量500mLのステンレス製ボトルです。重さは210g。',
    decide: '配送サービスは全国に対応。通常2営業日以内に発送します。土日祝日は発送を行いません。返品は未開封品に限り、到着後7日以内に受け付けます。',
    speech: 'こんにちは。今日は、新しいモデルを試してみましょう。'
  };
  function formValue() { return state.values[key()] ?? examples[state.demo] ?? ''; }
  function formDemo() {
    const labels = state.demo === 'speech' ? ['音声合成','テキストから音声を作ります。','読み上げるテキスト','音声を生成'] : state.demo === 'decide' ? ['文書をチェック','チェック項目ごとに、文書の内容を確認します。','判定する文書','チェックする'] : state.demo === 'lab' ? ['抽出ラボ','テキストから項目を取り出す流れを試します。','抽出するテキスト','抽出する'] : ['スペックメモ','商品説明から、スペックをまとめます。','商品説明','スペックを抽出'];
    const result = state.results[key()];
    return `${platform === 'ios' ? '' : intro(labels[0], labels[1])}<div class="demo-layout"><div><div class="field-label"><label for="demo-input">${labels[2]}</label><button type="button" data-action="example">例を使う</button></div><textarea id="demo-input" class="input-area" data-field="draft" spellcheck="false">${escape(formValue())}</textarea>${state.demo === 'lab' ? '<p class="caption">抽出する項目：商品名、容量、素材、重量</p>' : ''}${state.demo === 'decide' ? '<p class="caption">確認項目：全国配送 / 翌日発送 / 開封後の返品</p>' : ''}<button type="button" class="primary-button" data-action="run">${icon(state.demo === 'speech' ? 'irodori' : 'gliner')}${labels[3]}</button><p class="inline-feedback" id="feedback" role="status"></p></div><div><div class="result-heading"><span>${result ? '結果' : '結果のプレビュー'}</span>${result ? '<span class="preview-tag">サンプル出力</span>' : ''}</div>${result ? renderResult() : '<div class="result-placeholder">実行すると、ここに結果が表示されます。</div>'}</div></div><p class="sample-message">入力を編集できますが、この試作の出力は「例を使う」の固定サンプルです。モデルは実行しません。</p>`;
  }
  function renderResult() {
    if (state.demo === 'speech') return `<div class="result-card"><h3>音声のプレビュー</h3><svg viewBox="0 0 300 60" style="width:100%;height:60px" aria-label="音声波形のイメージ" role="img">${Array.from({length:48}, (_,i) => { const h = 8 + (i * 37 % 43); return `<rect x="${i*6+6}" y="${30-h/2}" width="3" height="${h}" rx="1.5" fill="#5b8b6b"/>`; }).join('')}</svg><p class="caption">波形の表示例です。このデザイン案では音声は再生しません。</p></div>`;
    const fields = state.demo === 'decide' ? [['全国へ配送できる','はい'],['翌日に発送される','文書からは不明'],['開封後も返品できる','いいえ']] : [['容量','500 mL'],['素材','ステンレス'],['重量','210 g'],['保温 / 保冷','6時間 / 12時間']];
    return `<div class="result-card"><h3>${state.demo === 'decide' ? 'チェック結果' : 'Aero Bottle 500'}</h3>${fields.map(([label,value])=>`<div class="result-item"><span>${label}</span><strong>${value}</strong></div>`).join('')}<p class="caption">${state.demo === 'decide' ? '入力例に対応する表示サンプル' : '出典：商品説明の入力例'}</p></div>`;
  }
  function art(kind) {
    const base = '<rect width="220" height="150" fill="#e7e9de"/><path d="M0 105 220 85v65H0Z" fill="#d4d6c6"/>';
    const shapes = kind === 'chair' ? '<rect x="72" y="33" width="65" height="58" rx="9" fill="#8e7956"/><path d="M72 85h76v15H72Z" fill="#aa9169"/><path d="M77 100v35m63-35v35" stroke="#776244" stroke-width="6"/>' : kind === 'cup' ? '<path d="M82 51h60v58q-30 20-60 0Z" fill="#f7f7ef"/><path d="M142 61h15q19 23-15 30" stroke="#f7f7ef" stroke-width="9" fill="none"/><ellipse cx="112" cy="51" rx="30" ry="7" fill="#bec6b4"/>' : '<path d="M108 100V47m0 24q-46-35-39-51 40-6 39 51m0 13q47-33 42-48-40-8-42 48" stroke="#759578" stroke-width="3" fill="#8ba68a"/><path d="m80 93 9 41h42l9-41Z" fill="#b99679"/>';
    return `<svg viewBox="0 0 220 150" role="img" aria-label="${kind === 'chair' ? '木の椅子' : kind === 'cup' ? '白いカップ' : '観葉植物'}のイラスト">${base}${shapes}</svg>`;
  }
  function imageDemo() {
    const finder = state.demo === 'finder';
    const result = state.results[key()];
    const query = state.values[key()] ?? (finder ? '白いカップ' : '木の椅子');
    return `<div class="image-demo">${platform === 'ios' ? '' : intro(names[state.demo], finder ? 'カメラに映るものを、言葉で探します。' : '画像の中から、言葉に近いものを探します。')}<label class="field-label" for="image-query">${finder ? '探したいもの' : 'どんな画像を探しますか？'}</label><input id="image-query" class="query-input" data-field="draft" value="${escape(query)}"><button type="button" class="primary-button" data-action="search-images">${icon(finder ? 'camera' : 'search')}${finder ? 'カメラの表示例を見る' : 'この画像から探す'}</button><div class="result-heading"><span>${result ? '検索結果の表示例' : 'サンプルの画像'}</span><span>3枚</span></div><div class="art-grid">${['chair','cup','plant'].map(kind => `<div class="art-tile ${result && kind === (finder ? 'cup':'chair') ? 'match':''}">${art(kind)}${result && kind === (finder ? 'cup':'chair') ? '<span class="match-label">一致する画像</span>':''}<p>${kind==='chair'?'木の椅子':kind==='cup'?'白いカップ':'観葉植物'}</p></div>`).join('')}</div><p class="sample-message">${finder ? 'カメラは使用しません。検出対象を示す固定の画面例です。' : '画像と結果は固定サンプルです。「木の椅子」の検索結果を示します。'}</p></div>`;
  }
  function chatDemo() {
    const result = state.results[key()];
    const work = state.demo === 'work';
    return `<div class="chat-workspace"><div class="chat-transcript">${result ? `<div class="demo-meta">${glyph(current())}<span>${current().name} · 4bit</span><span class="ready-badge">準備完了</span></div><div class="chat-bubble">${escape(result.input)}</div><div class="chat-answer"><span class="model-sub">${current().name} · <span class="preview-tag">サンプル出力</span></span>${work ? '調査の結果を、この領域にまとめます。\n\n参照したページや、実行した手順を回答と一緒に確認できる構成です。\n\nこのデザイン案ではWeb検索は行わず、固定の表示例を示しています。' : 'オンデバイスAIは、端末の中でモデルを動かして処理する仕組みです。\n\nたとえば、手元の文章から必要な情報を抽出したり、端末にある画像を言葉で探したりできます。\n\nこの画面では、選んだモデルと会話しながら、その応答を確かめられます。'}</div>` : `<div class="chat-intro">${glyph(current())}<h2 class="demo-title">${work ? '何を調べましょう？' : 'このモデルと、話してみる。'}</h2><p>${current().name} · ${current().precision}<br>${work ? 'Webを使うタスクを、選んだモデルに依頼します。' : 'モデルは選択済みです。そのままメッセージを送れます。'}</p><div class="chat-examples">${(work ? ['Core AIの特徴を調べて','参考資料をまとめて'] : ['オンデバイスAIとは？','文章の要約を試したい']).map(s=>`<button type="button" data-action="chat-example" data-value="${s}">${s}</button>`).join('')}</div></div>`}</div><div class="chat-compose"><label class="sr-only" for="chat-input">メッセージ</label><textarea id="chat-input" data-field="draft" placeholder="${work ? '依頼したいことを書く…' : 'メッセージを入力…'}">${escape(state.values[key()] ?? '')}</textarea><div class="compose-bottom"><span>${current().name} · ${names[state.demo]}</span><button type="button" class="send-button" data-action="send" aria-label="メッセージを送信">${icon('arrow')}</button></div></div><p class="sample-message">送信後は固定のサンプル回答を表示します。入力内容の送信やモデル推論は行いません。</p><p class="inline-feedback" id="feedback" role="status"></p></div>`;
  }
  function accessPanel() {
    if (canExecute()) return '';
    if (state.demo.startsWith('layout-')) return '<div class="demo-access" id="demo-access"><p>配置確認用の仮デモです。実行機能はありません。</p></div>';
    const m = current();
    const download = state.downloads.get(m.id);
    if (download?.phase === 'downloading') return `<div class="demo-access" id="demo-access"><strong data-access-progress>ダウンロード中 ${download.progress}%</strong><p>完了するまでは解説を閲覧できます。</p><progress class="access-progress" max="100" value="${download.progress}" aria-label="ダウンロード進捗"></progress><button type="button" class="secondary-button" data-action="cancel-download" data-model="${m.id}">キャンセル</button></div>`;
    const failed = download?.phase === 'failed';
    return `<div class="demo-access" id="demo-access"><strong>${failed ? 'ダウンロードできませんでした' : '解説を閲覧しています'}</strong><p>${failed ? '接続を確認して再試行してください。' : 'このデモを使うには、モデルのダウンロードが必要です。'}</p><button type="button" class="primary-button" data-action="download" data-model="${m.id}">${icon('download')}${failed ? '再試行' : 'ダウンロード'} · ${m.size}</button></div>`;
  }
  function demoContent() {
    if (state.demo.startsWith('layout-')) return '<div class="result-placeholder">デモの操作領域</div>';
    if (['chat','work'].includes(state.demo)) return chatDemo();
    if (['images','finder'].includes(state.demo)) return imageDemo();
    return formDemo();
  }
  function demoBody() {
    if (platform === 'ios') {
      const enabled = canExecute();
      return `${intro(demoDefinition(state.demo).title, '')}${accessPanel()}<fieldset class="demo-controls" ${enabled ? '' : 'disabled aria-describedby="demo-access"'}><legend class="sr-only">${demoDefinition(state.demo).title}の操作</legend>${demoContent()}</fieldset>`;
    }
    if (!state.installed.has(state.selected)) return prepare();
    return demoContent();
  }
  function detail() {
    if (!state.selected) return `<section class="desktop-detail" aria-label="モデル未選択"><div class="detail-toolbar"><span class="toolbar-name">モデルライブラリ</span><span class="muted" style="font-size:11px">まだ選択されていません</span></div><div class="library-empty"><div class="library-mark">${icon('library')}</div><h2>モデルを選んで、試す。</h2><p>左の一覧から気になるモデルを選ぶと、<br>そのモデルを使うデモがここに開きます。</p><div class="library-path"><strong>モデルを選択</strong><span aria-hidden="true">→</span><span>用途を試す</span><span aria-hidden="true">→</span><span>結果を確認</span></div><p class="library-foot">変換条件や対応環境は、モデル情報から確認できます。</p></div></section>`;
    const m = current();
    return `<section class="desktop-detail" aria-label="${m.name}のデモ"><div class="detail-toolbar"><button type="button" class="back-button" data-action="back">${icon('back')}モデル</button><div><span class="toolbar-name">${m.name}</span><span class="toolbar-version">${m.precision}</span></div>${state.installed.has(m.id) ? switcher() : ''}${modelInfoButton()}</div><div class="detail-scroll">${demoBody()}</div></section>`;
  }
  function render(focus = false) {
    if (platform === 'macos') {
      document.getElementById('desktop-window').dataset.view = state.selected ? 'demo' : 'library';
      app.innerHTML = sidebar() + detail();
    } else {
      app.innerHTML = state.selected ? `<div class="app-toolbar"><button type="button" class="back-button" data-action="back">${icon('back')}Models</button><span class="nav-title" aria-hidden="true">${current().name}</span>${modelInfoButton()}</div><div class="phone-scroll">${demoBody()}</div>` : library();
    }
    if (platform === 'ios') {
      document.querySelector('.phone')?.classList.toggle('large-text', state.largeText);
      app.classList.remove('is-scrolled');
      if (!state.selected) {
        const scroll = app.querySelector('.phone-scroll');
        scroll.scrollTop = state.libraryOffset;
        app.classList.toggle('is-scrolled', scroll.scrollTop > 65);
      }
    }
    if (focus) focusHeading();
  }
  function backToModels() {
    if (platform !== 'ios') { cancelTimer(); state.selected = null; state.demo = null; render(true); return; }
    const previousModel = state.selected;
    const previousDemo = state.demo;
    cancelTimer(); state.selected = null; state.demo = null; state.searchActive = false; render();
    const tile = app.querySelector(`[data-action="open-demo"][data-model="${previousModel}"][data-demo="${previousDemo}"]`);
    tile?.focus({preventScroll:true});
  }
  function cancelTimer() { clearTimeout(state.timer); state.timer = null; state.preparing = false; }
  function select(id, demoID) {
    const m = getModel(id);
    if (!m) return;
    if (!available(m)) { showDialog(m.name, `<p>${m.reason}</p><p>このiPhoneで使えるモデルを一覧から選んでください。</p><button type="button" class="primary-button" data-action="close-dialog">一覧に戻る</button>`); return; }
    if (demoID && !modelDemos(m).includes(demoID)) return;
    if (platform === 'ios' && !state.selected) state.libraryOffset = app.querySelector('.phone-scroll')?.scrollTop ?? 0;
    state.searchActive = false;
    cancelTimer(); state.selected = id; state.demo = demoID ?? modelDemos(m)[0]; render(true); announce(`${m.name}を選択しました。`);
  }
  function showDialog(title, html, confirmation = false) {
    dialog.classList.toggle('confirmation-dialog', confirmation);
    dialog.setAttribute('role', confirmation ? 'alertdialog' : 'dialog');
    const heading = document.getElementById('dialog-title');
    heading.textContent = title;
    heading.tabIndex = -1;
    document.getElementById('dialog-content').innerHTML = html;
    if (confirmation) dialog.setAttribute('aria-describedby', 'delete-explanation');
    else dialog.removeAttribute('aria-describedby');
    if (!dialog.open) dialog.showModal();
    if (confirmation) heading.focus({preventScroll:true});
  }
  function showInfo() {
    const m = current();
    showDialog(m.name, `<p>${m.description}</p><dl><div><dt>分類</dt><dd>${platform === 'ios' ? modelClassification(m) : m.type}</dd></div><div><dt>モデル容量</dt><dd>${m.size}</dd></div><div><dt>変換版</dt><dd>${m.precision}</dd></div><div><dt>試せるデモ</dt><dd>${modelDemos(m).map(id=>platform === 'ios' ? demoDefinition(id).title : names[id]).join(' / ')}</dd></div><div><dt>必要なメモリ</dt><dd>検証条件とともに掲載</dd></div></dl><p class="note">この画面の容量・状態はデザイン用のサンプルです。正式版では配布バンドルの容量、確認済み端末、検証記録へのリンクを表示します。</p><button type="button" class="primary-button" data-action="close-dialog">デモに戻る</button>`);
  }
  function runSample() {
    const value = document.getElementById('demo-input').value;
    state.values[key()] = value;
    if (!value.trim()) { document.getElementById('feedback').textContent = 'テキストを入力するか、「例を使う」を選んでください。'; return; }
    state.results[key()] = true; render(); announce('固定のサンプル結果を表示しました。');
    app.querySelector('.result-card')?.scrollIntoView({block:'nearest',behavior:'auto'});
  }
  document.addEventListener('toggle', event => {
    if (event.target.matches('details[data-explanation-key]') && event.target.isConnected) {
      state.explanations[event.target.dataset.explanationKey] = event.target.open;
    }
  }, true);
  document.addEventListener('scroll', event => {
    if (platform === 'ios' && event.target.matches?.('.phone-scroll')) {
      app.classList.toggle('is-scrolled', event.target.scrollTop > 65);
    }
  }, true);
  document.addEventListener('focusin', event => {
    if (platform === 'ios' && event.target.matches('[data-search]')) activateSearch();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && event.target.matches('[data-search]')) {
      event.preventDefault();
      if (platform === 'ios') cancelSearch(); else clearSearch();
    }
  });
  document.addEventListener('input', event => {
    if (event.target.matches('[data-search]')) {
      if (platform === 'ios') activateSearch();
      state.query = event.target.value;
      refreshSearch();
      if (platform === 'ios') app.querySelector('.phone-scroll').scrollTop = 0;
    }
    if (event.target.matches('[data-field="draft"]') && state.selected && (platform !== 'ios' || canExecute())) state.values[key()] = event.target.value;
  });
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-action]');
    if (!button || button.disabled) return;
    const executionActions = new Set(['example','run','search-images','chat-example','send']);
    if (platform === 'ios' && executionActions.has(button.dataset.action) && !canExecute()) return;
    switch (button.dataset.action) {
      case 'select': select(button.dataset.model); break;
      case 'open-demo': select(button.dataset.model, button.dataset.demo); break;
      case 'back': backToModels(); break;
      case 'reset': cancelTimer(); cancelDownloads(); state.selected = null; state.demo = null; state.query = ''; state.installed = new Set(['gliner','qwen']); state.values = {}; state.results = {}; state.explanations = {}; state.scale = false; state.searchActive = false; state.libraryOffset = 0; if (dialog.open) dialog.close(); render(true); announce('初期画面に戻りました。'); break;
      case 'clear-search':
      case 'clear-search-field': clearSearch(); break;
      case 'cancel-search': cancelSearch(); break;
      case 'preview-type': state.largeText = !state.largeText; button.setAttribute('aria-pressed', state.largeText); document.querySelector('.phone')?.classList.toggle('large-text', state.largeText); announce(state.largeText ? '文字サイズ200%のレイアウト確認です。' : '標準の文字サイズに戻しました。'); break;
      case 'preview-demo': state.installed.add(button.dataset.model); select(button.dataset.model); break;
      case 'preview-prepare': if (platform === 'ios') previewDownload(); else { state.installed.delete('embedding'); select('embedding'); } break;
      case 'preview-ten': cancelTimer(); state.selected = null; state.demo = null; state.query = ''; state.searchActive = false; state.libraryOffset = 0; state.scale = true; render(true); break;
      case 'preview-unavailable': previewDownload(); select('embedding','images'); break;
      case 'request-delete': {
        const m = getModel(button.dataset.model);
        if (!m || !state.installed.has(m.id)) break;
        showDialog(`${m.name}を削除しますか？`, `<p id="delete-explanation">${m.size}のモデルファイルを削除します。デモの解説や保存した入力・結果は残ります。</p><div class="confirmation-actions"><button type="button" class="confirmation-cancel" data-action="close-dialog">キャンセル</button><button type="button" class="confirmation-delete" data-action="delete-model" data-model="${m.id}">削除</button></div>`, true);
        break;
      }
      case 'delete-model': {
        const id = button.dataset.model;
        if (platform !== 'ios' || !getModel(id)) break;
        state.installed.delete(id); dialog.close(); refreshAcquisition(id); announce('モデルを削除した表示です。解説は引き続き閲覧できます。'); break;
      }
      case 'preview-download-failure': if (platform === 'ios') previewDownload(true); break;
      case 'download': startDownload(button.dataset.model); break;
      case 'cancel-download': cancelDownload(button.dataset.model); break;
      case 'demo': if (modelDemos(current()).includes(button.dataset.demo)) { state.demo = button.dataset.demo; render(true); } break;
      case 'prepare': {
        if (state.preparing || !state.selected) return;
        state.preparing = true;
        const selected = state.selected;
        render();
        state.timer = setTimeout(() => { if (state.selected !== selected || !state.preparing) return; state.installed.add(selected); state.preparing = false; state.timer = null; render(true); announce('準備完了の表示例です。デモを開きました。'); }, 1600);
        break;
      }
      case 'cancel-prepare': cancelTimer(); render(true); announce('準備をキャンセルしました。'); break;
      case 'use-local': state.installed.add(state.selected); render(true); announce('手元のモデルを指定した後の表示例です。'); break;
      case 'local': showDialog('ローカルモデルを開く', `<p>手元の変換済みモデルを、このライブラリに登録します。</p><div class="result-card" style="margin-top:18px"><div class="demo-meta">${icon('folder')}gliner-coreai /</div><p>GLiNER2.5-multi · スペックメモ</p></div><p class="note">操作例としてサンプルのフォルダを表示しています。実ファイルは読み取りません。</p><button type="button" class="primary-button" data-action="open-local-sample">このサンプルを開く</button>`); break;
      case 'open-local-sample': dialog.close(); state.installed.add('gliner'); select('gliner'); break;
      case 'info': showInfo(); break;
      case 'about': showDialog('Core AI Samples', '<p>変換したモデルを選び、実際の用途を通して動作を確かめるためのサンプラーです。</p><p>このページは操作を確認するためのデザイン案です。保存や外部通信は行わず、再読み込みすると初期状態に戻ります。</p><button type="button" class="primary-button" data-action="close-dialog">閉じる</button>'); break;
      case 'close-dialog': dialog.close(); break;
      case 'example': state.values[key()] = examples[state.demo]; delete state.results[key()]; render(); document.getElementById('demo-input')?.focus(); break;
      case 'run': runSample(); break;
      case 'search-images': state.results[key()] = true; render(); announce('検索結果の固定サンプルを表示しました。'); break;
      case 'chat-example': state.values[key()] = button.dataset.value; render(); document.getElementById('chat-input')?.focus(); break;
      case 'send': {
        const value = document.getElementById('chat-input').value.trim();
        if (!value) { document.getElementById('feedback').textContent = 'メッセージを入力してください。'; return; }
        state.results[key()] = { input: value }; state.values[key()] = ''; render(); announce('サンプル回答を表示しました。'); break;
      }
    }
  });
  dialog.addEventListener('click', event => { if (event.target === dialog) { const bounds = dialog.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close(); } });
  render();
})();
