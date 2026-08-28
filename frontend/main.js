/* AI Mentor — логика чата. Дизайн перенесён из мокапа ai_mentor.html.
 * Бэкенд: Tauri-команды send_message / get_status / get_model_status /
 * pick_model_file / start_model_download / cancel_model_download /
 * get_settings / set_settings (см. src-tauri/src/lib.rs). */
(function () {
  "use strict";

  var tauri = window.__TAURI__ || null;
  var invoke = tauri && tauri.core ? tauri.core.invoke : null;
  var listen = tauri && tauri.event ? tauri.event.listen : null;
  var clipboard = tauri && tauri.clipboard ? tauri.clipboard : null;

  var body = document.body;
  var feed = document.getElementById("feed");
  var promptEl = document.getElementById("prompt");
  var btnSend = document.getElementById("btnSend");
  var btnNew = document.getElementById("btnNew");
  var historyEl = document.getElementById("history");
  var threadTitle = document.getElementById("threadTitle");
  var statusText = document.getElementById("statusText");
  var ctxChip = document.getElementById("ctxChip");
  var kbInfo = document.getElementById("kbInfo");
  var themeToggle = document.getElementById("themeToggle");
  var navToggle = document.getElementById("navToggle");
  var backdrop = document.getElementById("backdrop");

  /* ---------- окно настроек ---------- */
  var settingsModal = document.getElementById("settingsModal");
  var btnSettings = document.getElementById("btnSettings");
  var btnChangeModel = document.getElementById("btnChangeModel");
  var btnSaveSettings = document.getElementById("btnSaveSettings");
  var btnCloseSettings = document.getElementById("btnCloseSettings");
  var setName = document.getElementById("setName");
  var setPath = document.getElementById("setPath");
  var setTemp = document.getElementById("setTemp");
  var setMaxTok = document.getElementById("setMaxTok");
  var setNCtx = document.getElementById("setNCtx");
  var setError = document.getElementById("setError");
  var lastSettings = null;

  /* ---------- экран "модель не найдена" ---------- */
  var setupScreen = document.getElementById("setupScreen");
  var btnPickModel = document.getElementById("btnPickModel");
  var btnStartDownload = document.getElementById("btnStartDownload");
  var btnCancelDownload = document.getElementById("btnCancelDownload");
  var dlUrlInput = document.getElementById("dlUrl");
  var dlHint = document.getElementById("dlHint");
  var dlBox = document.getElementById("dlBox");
  var dlFill = document.getElementById("dlFill");
  var dlPercent = document.getElementById("dlPercent");
  var dlBytes = document.getElementById("dlBytes");
  var dlResumeNote = document.getElementById("dlResumeNote");
  var setupError = document.getElementById("setupError");

  /* ---------- состояние диалогов ---------- */
  var threads = [];        // {id, title, messages:[{role, text|answer, thinking?, sources?, error?}]}
  var activeId = null;
  var llmReady = false;    // реальная LLM вместо заглушки (из get_status)

  /* ---------- Markdown-рендеринг модельного текста (этап L) ----------
   * Парсер markdown-it (терпим к незакрытым конструкциям — обязателен
   * для потока) + строгая санитизация DOMPurify. Вставка innerHTML
   * разрешена ТОЛЬКО через renderModelHtml/applyModelHtml — это единственная
   * точка, где модельный текст превращается в HTML. Пользовательский ввод
   * и структурированные данные (источники, статусы, ошибки) по-прежнему
   * идут только через textContent/createTextNode. */
  var mdParser = null;
  var purify = window.DOMPurify || null;
  try {
    if (window.markdownit) {
      mdParser = window.markdownit({
        html: false,     // сырой HTML во входном тексте экранируется, а не вставляется
        breaks: true,    // одиночный \n -> <br> (ближайший эквивалент прежнего pre-wrap)
        linkify: false   // не превращаем похожие на URL куски текста в <a>
      });
    }
  } catch (e) { mdParser = null; }

  // Возвращает санитизированный HTML или null, если библиотеки не загрузились
  // (тогда вызывающий обязан упасть в textContent-фолбэк — сырой текст).
  function renderModelHtml(text) {
    if (!mdParser || !purify) return null;
    var dirty = mdParser.render(String(text));
    var clean = purify.sanitize(dirty);
    return typeof clean === "string" ? clean : null;
  }

  // Добавляет кнопки копирования ко всем <pre><code> в элементе
  function injectCodeCopyButtons(rootEl) {
    var pres = rootEl.querySelectorAll("pre");
    pres.forEach(function (pre) {
      if (pre.querySelector(".code-copy-btn")) return; // уже есть
      var btn = document.createElement("button");
      btn.className = "code-copy-btn";
      btn.textContent = "copy";
      btn.title = "Скопировать код";
      btn.addEventListener("click", function () {
        var codeEl = pre.querySelector("code");
        var codeText = codeEl ? codeEl.textContent : pre.textContent;
        if (clipboard && clipboard.writeText) {
          clipboard.writeText(codeText).then(function () {
            btn.textContent = "скопировано";
            btn.classList.add("copied");
            setTimeout(function () {
              btn.textContent = "copy";
              btn.classList.remove("copied");
            }, 1000);
          }).catch(function () {
            fallbackCopy(codeText, btn);
          });
        } else {
          fallbackCopy(codeText, btn);
        }
      });
      pre.style.position = "relative"; // для absolute позиционирования кнопки
      pre.appendChild(btn);
    });
  }

  function fallbackCopy(text, btn) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      btn.textContent = "скопировано";
      btn.classList.add("copied");
      setTimeout(function () {
        btn.textContent = "copy";
        btn.classList.remove("copied");
      }, 1000);
    } catch (e) {
      btn.textContent = "ошибка";
      setTimeout(function () {
        btn.textContent = "copy";
      }, 1000);
    }
    document.body.removeChild(ta);
  }

  function applyModelHtml(el, text) {
    var html = renderModelHtml(text);
    if (html === null) { el.textContent = String(text); return; }
    el.innerHTML = html;
    injectCodeCopyButtons(el);
  }

  function nowTime() {
    var d = new Date();
    return ("0" + d.getHours()).slice(-2) + ":" + ("0" + d.getMinutes()).slice(-2);
  }

  function activeThread() {
    return threads.find(function (t) { return t.id === activeId; }) || null;
  }

  function newThread(silent) {
    var t = {
      id: Date.now(),
      title: "Новый диалог",
      messages: [],
      dayLabel: "сегодня"
    };
    threads.unshift(t);
    activeId = t.id;
    renderHistory();
    renderFeed();
    if (!silent) promptEl.focus();
  }

  /* ---------- рендер истории в сайдбаре ---------- */
  function renderHistory() {
    historyEl.innerHTML = "";
    var group = document.createElement("div");
    group.className = "group";
    var title = document.createElement("div");
    title.className = "group__title";
    title.textContent = "// диалоги";
    group.appendChild(title);

    threads.forEach(function (t) {
      var el = document.createElement("a");
      el.className = "thread" + (t.id === activeId ? " is-active" : "");
      var icon = document.createElement("span");
      icon.className = "thread__icon";
      icon.textContent = "◈";
      var name = document.createElement("span");
      name.className = "thread__name";
      name.textContent = t.title;
      var date = document.createElement("span");
      date.className = "thread__date";
      date.textContent = t.dateLabel || "";
      el.appendChild(icon); el.appendChild(name); el.appendChild(date);
      el.addEventListener("click", function () {
        activeId = t.id;
        renderHistory();
        renderFeed();
        body.classList.remove("nav-open");
      });
      group.appendChild(el);
    });
    historyEl.appendChild(group);
  }

  /* ---------- рендер ленты ---------- */
  function clearFeed() {
    feed.innerHTML = "";
  }

  function addDaySep(label) {
    var sep = document.createElement("div");
    sep.className = "day-sep";
    sep.textContent = label;
    feed.appendChild(sep);
  }

  function addUserMsg(text) {
    var wrap = document.createElement("article");
    wrap.className = "msg msg--user";
    var meta = document.createElement("div");
    meta.className = "msg__meta";
    meta.textContent = "ты · " + nowTime();
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(meta); wrap.appendChild(bubble);
    feed.appendChild(wrap);
  }

  /* ---------- живое потоковое сообщение (этап J) ----------
   * Токены приходят событиями "gen-token" (kind: think|answer) и/или
   * опросом get_gen_progress (снимок абсолютных значений). Текст ответа
   * появляется по мере генерации (эффект печатающейся машинки); ход
   * рассуждений стримится в свёрнутый think-блок, а его summary показывает
   * пульсирующий индикатор «модель размышляет…», пока идёт thinking-фаза. */
  var live = null;

  /* ---------- ЕДИНАЯ фабрика тела AI-сообщения (этап L3) ----------
   * И живой стрим, и финальный рендер строят DOM ОДНОЙ функцией:
   * think-details + контейнер ответа .md-body. Финальный рендер — это
   * не «другой рендер», а тот же applyModelHtml на том же .md-body
   * контейнере, вызванный с полным текстом вместо очередного чанка.
   * До L3 addAiAnswer строил msg__text без класса md-body, из-за чего
   * все CSS-правила `.md-body …` (таблицы, код-блоки, copy-кнопка)
   * переставали действовать сразу после завершения стрима
   * (см. logs/bug_repro.md, decisions.md L5). */
  function buildAiText(withCursor) {
    var text = document.createElement("div");
    text.className = "msg__text";

    // think-блок (скрыт, пока нет think-текста), свёрнут по умолчанию
    var thinkDetails = document.createElement("details");
    thinkDetails.className = "think";
    thinkDetails.hidden = true;
    var sum = document.createElement("summary");
    sum.className = "think__toggle";
    sum.textContent = "> раскрыть ход рассуждений";
    var thinkBody = document.createElement("div");
    thinkBody.className = "think__body md-body";
    thinkDetails.appendChild(sum);
    thinkDetails.appendChild(thinkBody);

    // контейнер ответа — ВСЕГДА .md-body (на нём висят стили Markdown)
    var ansEl = document.createElement("div");
    ansEl.className = "md-body";

    text.appendChild(thinkDetails);
    text.appendChild(ansEl);

    if (withCursor) {
      var cursor = document.createElement("span");
      cursor.className = "cursor-block";
      text.appendChild(cursor); // курсор всегда в конце активной области
    }

    return {
      root: text, summary: sum,
      thinkDetails: thinkDetails, thinkBody: thinkBody, ansEl: ansEl
    };
  }

  function startLiveMessage() {
    var wrap = document.createElement("article");
    wrap.className = "msg msg--ai";
    wrap.id = "live";
    var meta = document.createElement("div");
    meta.className = "msg__meta";
    meta.textContent = "ai-mentor · печатает…";
    var bodyEl = document.createElement("div");
    bodyEl.className = "msg__body";
    var prefix = document.createElement("span");
    prefix.className = "prefix";
    prefix.textContent = ">";

    var parts = buildAiText(true); // та же структура, что у финала (L3)

    bodyEl.appendChild(prefix); bodyEl.appendChild(parts.root);
    wrap.appendChild(meta); wrap.appendChild(bodyEl);
    feed.appendChild(wrap);

    live = {
      wrap: wrap, summary: parts.summary,
      thinkDetails: parts.thinkDetails, thinkBody: parts.thinkBody,
      ansEl: parts.ansEl,
      accThink: "", accAnswer: "", thinkingShown: false
    };
    scrollBottom();
  }

  function setThinkingIndicator(on) {
    if (!live) return;
    if (on && !live.thinkingShown) {
      live.summary.textContent = "> модель размышляет…";
      live.summary.classList.add("is-thinking");
      live.thinkingShown = true;
    } else if (!on && live.thinkingShown) {
      live.summary.textContent = "> раскрыть ход рассуждений";
      live.summary.classList.remove("is-thinking");
      live.thinkingShown = false;
    }
  }

  // Дельта из события "gen-token" (добавляется к накопленному). На КАЖДЫЙ
  // чанк накопленный текст заново прогоняется через markdown-it+DOMPurify —
  // форматирование появляется по ходу печати, а не в конце; markdown-it
  // корректно парсит незакрытые конструкции, поэтому ре-рендер на чанке
  // с оборванным "**" или "```" не рвёт вёрстку (проверено этапом L).
  function appendStreamDelta(kind, text) {
    if (!live || !text) return;
    if (kind === "think") {
      live.accThink += text;
      if (live.thinkDetails.hidden) live.thinkDetails.hidden = false;
      applyModelHtml(live.thinkBody, live.accThink);
      setThinkingIndicator(true);
    } else {
      live.accAnswer += text;
      setThinkingIndicator(false);
      applyModelHtml(live.ansEl, live.accAnswer);
    }
    scrollBottomThrottled();
  }

  // Абсолютный снимок из опроса get_gen_progress (запасной канал): применяем
  // только если он длиннее уже показанного, чтобы не конфликтовать с событиями.
  function applyStreamSnapshot(s) {
    if (!live || !s) return;
    if (s.thinking && s.thinking.length > live.accThink.length) {
      live.accThink = s.thinking;
      if (live.thinkDetails.hidden) live.thinkDetails.hidden = false;
      applyModelHtml(live.thinkBody, live.accThink);
      if (!s.answer || !s.answer.length) setThinkingIndicator(true);
    }
    if (s.answer && s.answer.length > live.accAnswer.length) {
      live.accAnswer = s.answer;
      setThinkingIndicator(false);
      applyModelHtml(live.ansEl, live.accAnswer);
    }
    scrollBottomThrottled();
  }

  function finalizeLive() {
    if (live) { live.wrap.remove(); live = null; }
  }

  var scrollPending = false;
  function scrollBottomThrottled() {
    if (scrollPending) return;
    scrollPending = true;
    requestAnimationFrame(function () {
      scrollPending = false;
      feed.scrollTop = feed.scrollHeight;
    });
  }

  /* опрос снимка генерации — запасной канал к событиям "gen-token" */
  var genPoller = null;
  function startGenPoller() {
    stopGenPoller();
    if (!invoke) return;
    genPoller = setInterval(function () {
      invoke("get_gen_progress").then(applyStreamSnapshot).catch(function () {});
    }, 300);
  }
  function stopGenPoller() {
    if (genPoller) { clearInterval(genPoller); genPoller = null; }
  }

  function errText(e) {
    return typeof e === "string" ? e : (e && e.message) || JSON.stringify(e);
  }

  function addAiAnswer(answer, sources, error, thinking) {
    var wrap = document.createElement("article");
    wrap.className = "msg msg--ai";
    var meta = document.createElement("div");
    meta.className = "msg__meta";
    meta.textContent = "ai-mentor · " + nowTime();
    var bodyEl = document.createElement("div");
    bodyEl.className = "msg__body";
    var prefix = document.createElement("span");
    prefix.className = "prefix";
    prefix.textContent = ">";

    // Тело строит ЕДИНАЯ фабрика со стримом (этап L3): тот же think-блок,
    // тот же контейнер ответа .md-body. Ниже — тот же applyModelHtml,
    // что в appendStreamDelta, просто с полным текстом вместо чанка.
    var parts = buildAiText(false);
    var text = parts.root;

    // Сворачиваемый блок хода рассуждений: свёрнут по умолчанию,
    // разворачивается кликом; финальный ответ ниже и всегда виден.
    if (thinking && String(thinking).trim()) {
      parts.thinkDetails.hidden = false;
      applyModelHtml(parts.thinkBody, String(thinking).trim());
    }

    if (error) {
      var err = document.createElement("div");
      err.className = "error";
      err.textContent = "ошибка: " + error;
      text.appendChild(err);
    } else {
      applyModelHtml(parts.ansEl, answer);
    }

    if (sources && sources.length) {
      // Сворачиваемый блок источников — тот же паттерн details/summary,
      // что у think-блока: свёрнут по умолчанию, разворачивается кликом.
      var src = document.createElement("details");
      src.className = "sources";
      var st = document.createElement("summary");
      st.className = "sources__toggle";
      st.textContent = "> источники (" + sources.length + ")";
      src.appendChild(st);
      var ol = document.createElement("ol");
      sources.forEach(function (h) {
        var li = document.createElement("li");
        var b = document.createElement("b");
        b.textContent = h.chunk_id;
        li.appendChild(b);
        li.appendChild(document.createTextNode(
          " · " + (h.topic || "-") + " · "
        ));
        var sc = document.createElement("span");
        sc.className = "score";
        sc.textContent = Number(h.score).toFixed(4);
        li.appendChild(sc);
        ol.appendChild(li);
      });
      src.appendChild(ol);
      text.appendChild(src);
    }

    bodyEl.appendChild(prefix); bodyEl.appendChild(text);
    wrap.appendChild(meta); wrap.appendChild(bodyEl);
    feed.appendChild(wrap);
    scrollBottom();
  }

  function scrollBottom() {
    feed.scrollTop = feed.scrollHeight;
  }

  function renderFeed() {
    clearFeed();
    var t = activeThread();
    if (!t) return;
    threadTitle.textContent = t.title;
    addDaySep(t.dayLabel);
    if (!t.messages.length) {
      addAiAnswer(
        llmReady
          ? "Наставник готов. Модель подключена локально (GGUF): спроси что-нибудь — " +
            "я найду релевантные фрагменты базы знаний и сгенерирую ответ по ним."
          : "Наставник готов к поиску. Ретрив по базе знаний настоящий, но модель не " +
            "загрузилась — проверь model_path в config.toml (или окно настроек): без неё " +
            "ответить нельзя.",
        []
      );
    }
    t.messages.forEach(function (m) {
      if (m.role === "user") addUserMsg(m.text);
      else addAiAnswer(m.answer, m.sources, m.error, m.thinking);
    });
  }

  /* ---------- отправка ---------- */
  var busy = false;

  function setBusy(b) {
    busy = b;
    btnSend.disabled = b;
    promptEl.disabled = b;
    document.querySelector(".composer__box").classList.toggle("disabled", b);
  }

  function send() {
    var q = promptEl.value.trim();
    if (!q || busy) return;
    if (!invoke) {
      // запуск вне Tauri (открыли index.html в браузере)
      alert("Фронтенд должен запускаться внутри Tauri-приложения.");
      return;
    }
    var t = activeThread() || (newThread(true), activeThread());
    if (t.title === "Новый диалог") {
      t.title = q.length > 34 ? q.slice(0, 33) + "…" : q;
      t.dateLabel = nowTime();
    }
    t.messages.push({ role: "user", text: q });

    promptEl.value = "";
    promptEl.style.height = "auto";
    renderHistory();
    renderFeed();

    setBusy(true);
    startLiveMessage();
    startGenPoller();

    invoke("send_message", { question: q })
      .then(function (reply) {
        stopGenPoller();
        finalizeLive();
        t.messages.push({
          role: "ai",
          answer: reply.answer,
          thinking: reply.thinking,
          sources: reply.sources
        });
        // финальный рендер канонический (абзацы + сворачиваемые источники),
        // он заменяет живое сообщение, чтобы история и стрим совпадали
        addAiAnswer(reply.answer, reply.sources, null, reply.thinking);
        setBusy(false);
        promptEl.focus();
      })
      .catch(function (e) {
        stopGenPoller();
        finalizeLive();
        var msg = errText(e);
        t.messages.push({ role: "ai", error: msg });
        addAiAnswer("", [], msg);
        setBusy(false);
        promptEl.focus();
      });
  }

  /* ---------- тема ---------- */
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    themeToggle.textContent =
      theme === "light" ? "◐ тема: светлая" : "◐ тема: тёмная";
    try { localStorage.setItem("aim_theme", theme); } catch (e) {}
  }

  /* ---------- статус ---------- */
  function loadStatus() {
    if (!invoke) {
      statusText.innerHTML = "<b>вне Tauri</b> · демо";
      return Promise.resolve();
    }
    return invoke("get_status")
      .then(function (s) {
        kbInfo.textContent =
          s.collection + ": " + s.points + " чанков · " + s.embedding_model;
        ctxChip.textContent = "top-k " + s.top_k;
        llmReady = !s.generator_stub;
        var label = s.generator_stub
          ? "модель не загружена" + (s.model_path_set ? "" : " (model_path пуст)")
          : "LLM локально";
        statusText.innerHTML = "";
        var dot = statusText.parentElement.querySelector(".dot");
        var b = document.createElement("b");
        b.textContent = label;
        statusText.appendChild(b);
        statusText.appendChild(document.createTextNode(" · ретрив онлайн"));
        if (dot) dot.remove(); // статус подтверждён — пульс больше не нужен
        // приветствие зависит от статуса генератора; обновить пустой диалог
        var t0 = activeThread();
        if (t0 && !t0.messages.length) renderFeed();
      })
      .catch(function (e) {
        statusText.parentElement.classList.add("err");
        statusText.innerHTML = "";
        var b = document.createElement("b");
        b.textContent = "Qdrant недоступен";
        statusText.appendChild(b);
        statusText.appendChild(document.createTextNode(
          " · сервис Qdrant поднимается приложением автоматически; перезапусти приложение"
        ));
      });
  }

  /* ---------- экран настройки модели ---------- */
  function unlockComposer() {
    promptEl.disabled = false;
    btnSend.disabled = false;
    promptEl.focus();
  }

  function fmtMiB(bytes) {
    return (bytes / (1024 * 1024)).toFixed(1);
  }

  function openSetup(st) {
    body.classList.add("setup-mode");
    setupScreen.hidden = false;
    setupError.hidden = true;
    dlBox.hidden = true;
    btnCancelDownload.hidden = true;
    btnPickModel.disabled = false;
    btnStartDownload.disabled = false;

    var url = (st.download_url || "").trim();
    dlUrlInput.value = url;
    if (!url) {
      btnStartDownload.disabled = true;
      dlHint.textContent =
        "кнопка скачивания недоступна: заполни model_download_url в config.toml и перезапусти приложение";
    } else {
      dlHint.textContent = st.sha256_set
        ? "после загрузки будет проверена контрольная сумма SHA-256"
        : "";
    }
  }

  function closeSetupAndEnterChat() {
    setupScreen.hidden = true;
    body.classList.remove("setup-mode");
    unlockComposer();
    loadStatus(); // обновить статус-бар и приветствие
  }

  /* прогресс загрузки: скорость считаем по дельтам событий */
  var lastProgressAt = 0;
  var lastProgressBytes = 0;
  var progressPoller = null;
  var downloadFinished = false;

  function stopPolling() {
    if (progressPoller) {
      clearInterval(progressPoller);
      progressPoller = null;
    }
  }

  function handleDownloadEvent(p) {
    if (downloadFinished) return;
    if (p.error) {
      downloadFinished = true;
      stopPolling();
      finishDownloadUI(false);
      setupError.hidden = false;
      setupError.textContent = "// ошибка загрузки\n" + p.error;
      return;
    }
    if (p.done) {
      downloadFinished = true;
      stopPolling();
      finishDownloadUI(true);
      dlPercent.textContent = "готово";
      dlFill.style.width = "100%";
      dlBytes.textContent = "модель сохранена, открываю чат…";
      closeSetupAndEnterChat();
      return;
    }
    if (p.resumed_from > 0 && dlResumeNote.textContent === "") {
      dlResumeNote.textContent =
        "докачка с " + fmtMiB(p.resumed_from) + " МиБ";
    }
    var now = Date.now();
    if (lastProgressAt && p.downloaded >= lastProgressBytes) {
      var dt = (now - lastProgressAt) / 1000;
      var speed = dt > 0 ? (p.downloaded - lastProgressBytes) / dt : 0;
      dlBytes.textContent =
        fmtMiB(p.downloaded) + (p.total ? " / " + fmtMiB(p.total) : "") +
        " МиБ · " + (speed / (1024 * 1024)).toFixed(1) + " МиБ/с";
    }
    lastProgressAt = now;
    lastProgressBytes = p.downloaded;

    if (p.total > 0) {
      var pct = Math.min(100, p.downloaded / p.total * 100);
      dlPercent.textContent = pct.toFixed(1) + "%";
      dlFill.style.width = pct.toFixed(2) + "%";
      dlBox.classList.remove("is-indeterminate");
    } else {
      dlPercent.textContent = "…";
      dlBox.classList.add("is-indeterminate");
    }
  }

  function startDownloadUI() {
    setupError.hidden = true;
    dlBox.hidden = false;
    dlResumeNote.textContent = "";
    dlFill.style.width = "0%";
    dlPercent.textContent = "0%";
    dlBytes.textContent = "соединение…";
    dlBox.classList.add("is-indeterminate");
    btnStartDownload.disabled = true;
    btnPickModel.disabled = true;
    btnCancelDownload.hidden = false;
    lastProgressAt = 0;
    lastProgressBytes = 0;
    downloadFinished = false;
    // запасной канал: опрос снимка прогресса; старый интервал всегда гасим
    stopPolling();
    if (invoke) {
      progressPoller = setInterval(function () {
        invoke("get_download_progress")
          .then(function (p) {
            if (p) handleDownloadEvent(p);
          })
          .catch(function () { /* событие ошибки придёт отдельно */ });
      }, 400);
    }
  }

  function finishDownloadUI(success) {
    dlBox.classList.remove("is-indeterminate");
    btnPickModel.disabled = success ? true : false;
    btnCancelDownload.hidden = true;
    if (!success) {
      btnStartDownload.disabled = false;
    } else {
      btnStartDownload.disabled = true;
    }
  }

  /* ---------- события ---------- */
  navToggle.addEventListener("click", function () { body.classList.toggle("nav-open"); });
  backdrop.addEventListener("click", function () { body.classList.remove("nav-open"); });

  btnNew.addEventListener("click", function () { newThread(false); });

  promptEl.addEventListener("input", function () {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 180) + "px";
  });

  promptEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  btnSend.addEventListener("click", send);

  themeToggle.addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme") === "light"
      ? "dark" : "light";
    applyTheme(cur);
  });

  btnPickModel.addEventListener("click", function () {
    if (!invoke) return;
    btnPickModel.disabled = true;
    invoke("pick_model_file")
      .then(function (st) {
        btnPickModel.disabled = false;
        if (!st) return; // диалог закрыт без выбора
        if (st.found) closeSetupAndEnterChat();
        else {
          setupError.hidden = false;
          setupError.textContent = "// файл не подошёл\nвыбранный путь не существует";
        }
      })
      .catch(function (e) {
        btnPickModel.disabled = false;
        setupError.hidden = false;
        var msg = errText(e);
        setupError.textContent = "// ошибка выбора файла\n" + msg;
      });
  });

  btnStartDownload.addEventListener("click", function () {
    if (!invoke) return;
    startDownloadUI();
    invoke("start_model_download").catch(function (e) {
      var msg = errText(e);
      finishDownloadUI(false);
      setupError.hidden = false;
      setupError.textContent = "// не удалось начать загрузку\n" + msg;
    });
  });

  btnCancelDownload.addEventListener("click", function () {
    if (!invoke) return;
    invoke("cancel_model_download").catch(function () { /* событие ошибки придёт само */ });
  });

  if (listen) listen("download-progress", function (ev) { handleDownloadEvent(ev.payload); });

  // Потоковая генерация ответа: каждый токен приходит событием (kind + текст).
  if (listen) listen("gen-token", function (ev) {
    var p = ev.payload;
    if (p) appendStreamDelta(p.kind, p.text);
  });

  /* ---------- окно настроек ---------- */
  function showSettingsError(msg) {
    setError.hidden = false;
    setError.textContent = msg;
  }

  function hideSettingsError() {
    setError.hidden = true;
    setError.textContent = "";
  }

  function fillSettings(s) {
    lastSettings = s;
    setName.textContent = s.model_name || "— имя недоступно —";
    setPath.textContent = s.model_path || "(model_path пуст)";
    setPath.title = setPath.textContent;
    setTemp.value = s.temperature;
    setMaxTok.value = s.max_tokens;
    updateCtxHint();
  }

  // Справочная строка n_ctx пересчитывается при правке max_tokens,
  // чтобы не показывать устаревшее значение до сохранения.
  function updateCtxHint() {
    if (!lastSettings) return;
    var mt = parseInt(String(setMaxTok.value), 10);
    var shown = isNaN(mt) ? lastSettings.max_tokens : mt;
    setNCtx.value = "n_ctx = " + lastSettings.n_ctx +
      " · RAG-промпт до ~2.8k + ответ " + shown;
  }
  setMaxTok.addEventListener("input", updateCtxHint);

  function openSettings() {
    if (!invoke) return;
    hideSettingsError();
    settingsModal.hidden = false;
    invoke("get_settings")
      .then(fillSettings)
      .catch(function (e) {
        var msg = errText(e);
        showSettingsError("// не удалось прочитать настройки\n" + msg);
      });
  }

  btnSettings.addEventListener("click", openSettings);

  btnCloseSettings.addEventListener("click", function () {
    settingsModal.hidden = true;
  });

  // Клик по затемнению вне бокса закрывает окно
  settingsModal.addEventListener("click", function (ev) {
    if (ev.target === settingsModal) settingsModal.hidden = true;
  });

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && !settingsModal.hidden) settingsModal.hidden = true;
  });

  btnSaveSettings.addEventListener("click", function () {
    if (!invoke) return;
    hideSettingsError();
    var temperature = parseFloat(String(setTemp.value).replace(",", "."));
    var maxTokens = parseInt(String(setMaxTok.value), 10);
    if (isNaN(temperature)) { showSettingsError("// ошибка\n temperature — число, например 0.7"); return; }
    if (isNaN(maxTokens)) { showSettingsError("// ошибка\nmax_tokens — целое число, например 5500"); return; }
    btnSaveSettings.disabled = true;
    invoke("set_settings", { temperature: temperature, maxTokens: maxTokens })
      .then(function (s) {
        btnSaveSettings.disabled = false;
        fillSettings(s);
        settingsModal.hidden = true;
      })
      .catch(function (e) {
        btnSaveSettings.disabled = false;
        var msg = errText(e);
        showSettingsError("// не сохранено\n" + msg);
      });
  });

  // Смена модели — тот же диалог выбора файла, что на стартовом экране.
  btnChangeModel.addEventListener("click", function () {
    if (!invoke) return;
    hideSettingsError();
    btnChangeModel.disabled = true;
    invoke("pick_model_file")
      .then(function (st) {
        btnChangeModel.disabled = false;
        if (!st) return; // диалог закрыт без выбора
        return invoke("get_settings").then(fillSettings).catch(function () {});
      })
      .catch(function (e) {
        btnChangeModel.disabled = false;
        var msg = errText(e);
        showSettingsError("// не удалось сменить модель\n" + msg);
      });
  });

  /* ---------- старт ---------- */
  var savedTheme = "dark";
  try { savedTheme = localStorage.getItem("aim_theme") || "dark"; } catch (e) {}
  applyTheme(savedTheme);

  newThread(true);
  loadStatus();

  // Модель готова -> чат; нет -> экран "модель не найдена" вместо чата.
  // Композер остаётся заблокированным до решения.
  if (!invoke) {
    unlockComposer(); // запуск вне Tauri: демо-режим фронтенда
  } else {
    invoke("get_model_status")
      .then(function (st) {
        if (st.found) unlockComposer();
        else openSetup(st);
      })
      .catch(function () {
        unlockComposer(); // ошибка чтения конфига проявится в send_message
      });
  }
})();
