'use strict';

/* =========================================================
 * Dengue risk self-assessment - front-end logic
 *
 * The questionnaire fields align strictly with the back-end contract
 * (app/schemas.py):
 *   age(0-110) / sex(F|M) / day_ill(0-14)
 *   symptoms      -- 14 items, each yes | no | unknown
 *   comorbidities -- 7 items, each yes | no | unknown
 *   exposure      -- 3 items, each yes | no | unknown (**takes no part in scoring**)
 *   language / notes
 *
 * "Don't know" is 0 in the model, the same as "no" (the SINAN training data's
 * 9=unknown is recorded as 0), so an unanswered form can be submitted as-is and
 * users are never forced to answer the symptom questions.
 *
 * Epidemiological exposure does not exist in the SINAN training data and has no
 * coefficient available, so the back end puts it in a separate rule-based channel,
 * exposure_context. The front end must present it differently too: it is not a
 * model score and must not be made to look like a gauge.
 * ========================================================= */

const LANGS = [
  { code: 'zh-CN', name: '简体中文' },
  { code: 'zh-TW', name: '繁體中文' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'pt', name: 'Português' },
];

const DEFAULT_LANG = 'zh-CN';

// Matches the back end's SYMPTOM_CODES / COMORB_CODES
const SYMPTOM_CODES = [
  'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS',
  'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N', 'LEUCOPENIA', 'LACO', 'DOR_RETRO',
];
const COMORB_CODES = [
  'DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE',
];
// Epidemiological exposure: goes down the back end's rule channel, not into the model
const EXPOSURE_CODES = ['FEVER_CLUSTER', 'CONFIRMED_CASE', 'OUTBREAK_TRAVEL'];

// Field categories with three-state answers (used by generic logic such as counting)
const TRI_KINDS = ['symptoms', 'comorbidities', 'exposure'];

// Smart interview (adaptive mode):
// - VOMITO / PETEQUIA_N are WHO warning signs and are asked of everyone (safety stage)
// - the adaptive stage asks at most ADAPTIVE_CAP questions
// - if /api/plan exceeds PLAN_TIMEOUT_MS or errors, silently fall back to the full
//   questionnaire
const SAFETY_CODES = ['VOMITO', 'PETEQUIA_N'];
const ADAPTIVE_STAGES = ['basic', 'safety', 'exposure', 'loop', 'notes'];
const ADAPTIVE_CAP = 12;
const PLAN_TIMEOUT_MS = 6000;
const PLAN_KIND_MAP = { symptom: 'symptoms', comorbidity: 'comorbidities' };

// The 5 non-binary features in the model contributions (code keeps its own name, no _x)
const NON_BINARY_FEATS = ['age', 'sex_f', 'day_ill', 'wk_sin', 'wk_cos'];

// The seven-step questionnaire structure
const STEPS = [
  { id: 'basic', kind: 'basic' },
  { id: 'common', kind: 'symptoms', codes: ['FEBRE', 'CEFALEIA', 'MIALGIA', 'ARTRALGIA', 'DOR_RETRO', 'DOR_COSTAS'] },
  { id: 'other', kind: 'symptoms', codes: ['NAUSEA', 'VOMITO', 'EXANTEMA', 'CONJUNTVIT', 'ARTRITE'] },
  { id: 'clinical', kind: 'symptoms', codes: ['PETEQUIA_N', 'LACO', 'LEUCOPENIA'] },
  { id: 'exposure', kind: 'exposure', codes: EXPOSURE_CODES },
  { id: 'history', kind: 'comorbidities', codes: COMORB_CODES },
  { id: 'notes', kind: 'notes' },
];

// The result page's three model metrics <-> the keys of explanations
const METRICS = ['dengue', 'worsening', 'severe'];

// Advice display order: seek care -> monitoring -> protection (matching the field order
// of the back end's Advice)
const ADVICE_ORDER = ['medical', 'monitoring', 'protection'];

// Chat: send back at most the 6 most recent history entries, question capped at 500
// characters (matching app/schemas.py)
const CHAT_HISTORY_MAX = 6;
const CHAT_QUESTION_MAX = 500;

/*
 * Destination lookup (the second user path) -- the /api/destination contract:
 *   request  { location: str(1..120), language }
 *   response { location, matched, endemicity, season_note, who_notices[],
 *              recent_findings[], sources[], advice{}, search_status,
 *              disclaimer, model_note }
 *
 * The back end **deliberately returns no score at all**: a country-level reference
 * table cannot support a 0-100 number. The front end must not invent one either:
 * this page has no gauge and no score, only qualitative badges.
 *
 * The three layers of information decrease in trustworthiness and must be labelled
 * separately:
 *   endemicity/season_note -> stable reference knowledge;
 *   who_notices            -> official WHO notices, **every one must show its real
 *                             date** (some are years old);
 *   recent_findings        -> results from the model's web search covering roughly the
 *                             last three months; may be empty.
 */
const DEST_LOCATION_MAX = 120;
const DEST_MIN_WAIT_MS = 900;
const ENDEMICITY_LEVELS = ['high', 'moderate', 'low', 'none', 'unknown'];

/* --------------------------------------------------------- Copy */

const I18N = {
  'zh-CN': {
    docTitle: '登革热风险自测',
    metaDescription: '回答症状与病史问题，获得登革热风险参考评分与防护建议。结果仅供参考，不构成医疗诊断。',
    langLabel: '选择语言',
    a11y: { progress: '问卷进度', chatLog: '对话记录' },
    hero: {
      title: '登革热<br />风险自测',
      features: ['登革热专用', '真实数据建模', '五种语言'],
      ctaTravelTitle: '我要去某个地方',
      ctaTravelDesc: '查看目的地的登革热情况',
      ctaSymptomsTitle: '我有症状',
      ctaSymptomsDesc: '回答几个问题，获得风险参考',
      privacyLink: '隐私说明',
    },
    sources: { label: '信息来源', who: '世界卫生组织', search: '网络检索', official: '官方' },
    dest: {
      title: '目的地登革热情况',
      sub: '看看你要去的地方目前已知的登革热情况。本页不提供个人风险评分。',
      back: '返回',
      inputLabel: '目的地',
      placeholder: '国家、地区或城市',
      submit: '查询',
      examplesLabel: '例如',
      examples: ['新加坡', '巴西', '泰国', '菲律宾'],
      emptyHint: '请先输入目的地。',
      loading: {
        steps: ['正在查询地区背景资料…', '正在检索近期报道…', '正在汇总结果…'],
      },
      errors: {
        network: '网络连接失败，请检查网络后重试。',
        server: '目的地查询服务暂时不可用，请稍后重试。',
        generic: '请求失败（状态码 {status}），请稍后重试。',
        badData: '服务返回的数据格式异常，请稍后重试。',
        retry: '重试',
      },
      resultTitle: '{location} 的登革热情况',
      notMatched: '参考表里没有匹配到「{location}」，因此不显示流行程度等级。下面查到的内容仍会照常列出。',
      layers: {
        reference: {
          title: '地区背景',
          tag: '稳定参考资料',
        },
        who: {
          title: '世界卫生组织通报',
          tag: '官方',
          empty: '这个目的地没有查到世界卫生组织通报。',
        },
        recent: {
          title: '近期报道',
          tag: '网络检索 · 近约三个月',
          empty: '没有检索到近期报道。',
        },
      },
      endemicity: {
        label: '流行程度',
        levels: { high: '高', moderate: '中等', low: '低', none: '无本地传播报告', unknown: '未知' },
      },
      seasonLabel: '季节性',
      status: {
        degraded: '本次未能获取近期信息，因此只显示地区背景与世界卫生组织两层内容。',
        disabled: '联网检索目前处于关闭状态，因此只显示地区背景与世界卫生组织两层内容。',
      },
      toSymptoms: '身体不舒服？去做症状自测',
    },
    travel: {
      label: '旅行背景',
      place: '目的地：{location}',
      level: '流行程度：{level}',
      notMatched: '参考表中没有这个地点',
      offer: '要出门旅行吗？',
      offerLink: '查询目的地的登革热情况',
    },
    privacy: {
      title: '隐私说明',
      intro: '这个工具在设计上就不需要知道你是谁。以下是它对你的数据所做的全部事情：',
      items: [
        '无需注册、无需登录，不收集姓名、电话、邮箱等任何联系方式。',
        '你的作答不会与任何身份信息关联。',
        '「补充说明」里自由填写的文字不会被保存。',
        '去标识化的作答与评分会被记录，用于改进模型校准。',
        '不出售、也不与第三方共享你的数据。',
        '启用 AI 建议时，你的作答摘要会发送给第三方大语言模型接口进行处理。',
        '评估结果不是医疗诊断。',
      ],
      foot: '本说明描述本服务当前的实际运作方式；今后如有变化会同步更新。',
      close: '知道了',
    },
    brand: '🦟 登革热风险自测',
    stepCounter: '第 {cur} 步 / 共 {total} 步',
    steps: {
      basic: { title: '基本信息' },
      common: { title: '常见症状', },
      other: { title: '其他症状', },
      clinical: { title: '出血与化验', sub: '根据医生检查结果' },
      exposure: { title: '周围环境与暴露' },
      history: { title: '既往病史' },
      notes: { title: '补充说明', },
    },
    fields: {
      age: '年龄', ageUnit: '岁',
      sex: '性别', sexF: '女', sexM: '男',
      dayIll: '症状已持续', dayIllUnit: '天',
      dayIllZero: '今天刚开始', dayIllNone: '尚无症状',
    },
    answers: { yes: '有', no: '无', unknown: '不知道' },
    answered: '已答 {n}/{total}',
    notes: {
      placeholder: '例如：三天前开始发烧，眼睛后面很胀，昨天开始牙龈有点出血…',
    },
    nav: { prev: '上一步', next: '下一步', submit: '提交评估', back: '返回' },
    hints: { sexRequired: '请选择性别后继续' },
    mode: {
      adaptive: '智能问诊（推荐）',
      classic: '完整问卷',
      switch: '选择问诊方式',
    },
    adaptive: {
      stageSafety: '安全筛查',
      stageLoop: '智能提问',
      questionCounter: '第 {n} 题',
      continueFull: '补充更多问题（转完整问卷）',
    },
    loading: {
      steps: ['正在整理你的作答…', '模型评估中…', '正在生成个性化建议…'],
    },
    result: {
      title: '评估结果',
      dengue: '登革热可能性',
      severe: '重症风险',
      worsening: '病情加重风险',
      levels: { low: '低风险', medium: '中风险', high: '高风险' },
      advice: { medical: '🏥 就医提示', monitoring: '📋 居家监测', protection: '🛡️ 防蚊与防护' },
      explain: {
        why: '为什么是这个分数？',
        title: '{metric} · 影响最大的因素',
        up: '推高评分',
        down: '拉低评分',
        empty: '这一项暂时没有可展示的贡献项。',
      },
      exposure: {
        label: '周围暴露情况',
        levels: { low: '低', medium: '中', high: '高' },
        factors: '触发因素：{list}',
        none: '你没有报告周围有相关的暴露情况。',
        sep: '、',
      },
      warning: {
        title: '出现登革热警示征象',
        body: '你报告了 {signs}，这属于世界卫生组织列出的登革热警示征象。无论上方评分高低，都建议尽快就医评估。',
        sep: '、',
      },
      restart: '重新评估',
      home: '返回首页',
    },
    errors: {
      title: '评估失败',
      network: '网络连接失败，请检查网络后重试。',
      validation: '提交的信息有误，请返回检查后重试。',
      upstream: '模型服务暂时不可用，请稍后重试。',
      server: '服务器开小差了，请稍后重试。',
      generic: '请求失败（状态码 {status}），请稍后重试。',
      badData: '服务返回的数据格式异常，请稍后重试。',
      retry: '重试',
      back: '返回修改',
    },
    disclaimer: '本结果仅供参考，不构成医疗诊断。如有不适请及时就医。',
    symptoms: {
      FEBRE: { label: '发热', desc: '体温超过 37.5°C' },
      MIALGIA: { label: '肌肉痛', desc: '全身肌肉酸痛' },
      CEFALEIA: { label: '头痛', desc: '持续性头痛' },
      EXANTEMA: { label: '皮疹', desc: '皮肤出现红色斑疹' },
      VOMITO: { label: '呕吐', desc: '已经吐出胃内容物' },
      NAUSEA: { label: '恶心', desc: '想吐但没有吐出来' },
      DOR_COSTAS: { label: '背痛', desc: '腰背部疼痛' },
      CONJUNTVIT: { label: '结膜炎', desc: '眼睛发红、有分泌物' },
      ARTRITE: { label: '关节炎', desc: '关节红肿发热' },
      ARTRALGIA: { label: '关节痛', desc: '关节疼痛但没有红肿' },
      PETEQUIA_N: { label: '皮肤瘀点', desc: '针尖大小的红点，按压不褪色' },
      LEUCOPENIA: { label: '白细胞减少', desc: '血常规显示白细胞偏低（需验血）' },
      LACO: { label: '束臂试验阳性', desc: '医生用血压计加压后皮肤出现出血点' },
      DOR_RETRO: { label: '眼后痛', desc: '眼球后方疼痛，转动眼球时加重' },
    },
    comorbidities: {
      DIABETES: { label: '糖尿病', desc: '已确诊的糖尿病' },
      HEMATOLOG: { label: '血液疾病', desc: '如贫血、凝血障碍、血小板疾病' },
      HEPATOPAT: { label: '肝病', desc: '如肝炎、肝硬化' },
      RENAL: { label: '肾脏病', desc: '如慢性肾病、肾功能不全' },
      HIPERTENSA: { label: '高血压', desc: '已确诊的高血压' },
      ACIDO_PEPT: { label: '消化性溃疡', desc: '胃溃疡或十二指肠溃疡' },
      AUTO_IMUNE: { label: '自身免疫疾病', desc: '如红斑狼疮、类风湿关节炎' },
    },
    exposure: {
      FEVER_CLUSTER: { label: '周围发热病例增多', desc: '家人、同事、同学或邻居中近期发热的人明显变多' },
      CONFIRMED_CASE: { label: '身边有确诊病例', desc: '家庭、工作场所或社区里有人被确诊登革热' },
      OUTBREAK_TRAVEL: { label: '暴发地区居住或旅行', desc: '近期到访过、或正居住在登革热暴发地区' },
    },
    features: {
      age: '年龄',
      sex_f: '女性',
      day_ill: '症状持续天数',
      wk_sin: '季节性周期（分量一）',
      wk_cos: '季节性周期（分量二）',
    },
    chat: {
      title: '追问一下',
      inputLabel: '你的问题',
      placeholder: '输入你的问题…',
      send: '发送',
      typing: 'AI 正在回复…',
      you: '你',
      assistant: 'AI 助手',
      note: '回答由 AI 生成，仅供参考，不作为医疗诊断依据。',
      privacyLink: '隐私说明',
      error: 'AI 助手暂时没能回答，请稍后重试。',
      retry: '重试',
      chips: [
        '这些分数到底是什么意思？',
        '我现在需要去看医生吗？',
        '我应该特别留意哪些危险信号？',
        '怎么保护家里其他人？',
        '我很难受，为什么重症风险还是低？',
        '这算是确诊吗？',
      ],
    },
  },

  'zh-TW': {
    docTitle: '登革熱風險自測',
    metaDescription: '回答症狀與病史問題，獲得登革熱風險參考評分與防護建議。結果僅供參考，不構成醫療診斷。',
    langLabel: '選擇語言',
    a11y: { progress: '問卷進度', chatLog: '對話紀錄' },
    hero: {
      title: '登革熱<br />風險自測',
      features: ['登革熱專用', '真實數據建模', '五種語言'],
      ctaTravelTitle: '我要去某個地方',
      ctaTravelDesc: '查看目的地的登革熱情況',
      ctaSymptomsTitle: '我有症狀',
      ctaSymptomsDesc: '回答幾個問題，取得風險參考',
      privacyLink: '隱私說明',
    },
    sources: { label: '資訊來源', who: '世界衛生組織', search: '網路搜尋', official: '官方' },
    dest: {
      title: '目的地登革熱情況',
      sub: '看看你要去的地方目前已知的登革熱情況。本頁不提供個人風險評分。',
      back: '返回',
      inputLabel: '目的地',
      placeholder: '國家、地區或城市',
      submit: '查詢',
      examplesLabel: '例如',
      examples: ['新加坡', '巴西', '泰國', '菲律賓'],
      emptyHint: '請先輸入目的地。',
      loading: {
        steps: ['正在查詢地區背景資料…', '正在搜尋近期報導…', '正在彙整結果…'],
      },
      errors: {
        network: '網路連線失敗，請檢查網路後重試。',
        server: '目的地查詢服務暫時無法使用，請稍後重試。',
        generic: '請求失敗（狀態碼 {status}），請稍後重試。',
        badData: '服務回傳的資料格式異常，請稍後重試。',
        retry: '重試',
      },
      resultTitle: '{location} 的登革熱情況',
      notMatched: '參考表中沒有比對到「{location}」，因此不顯示流行程度等級。下面查到的內容仍會照常列出。',
      layers: {
        reference: {
          title: '地區背景',
          tag: '穩定參考資料',
        },
        who: {
          title: '世界衛生組織通報',
          tag: '官方',
          empty: '這個目的地沒有查到世界衛生組織通報。',
        },
        recent: {
          title: '近期報導',
          tag: '網路搜尋 · 近約三個月',
          empty: '沒有搜尋到近期報導。',
        },
      },
      endemicity: {
        label: '流行程度',
        levels: { high: '高', moderate: '中等', low: '低', none: '無本地傳播通報', unknown: '未知' },
      },
      seasonLabel: '季節性',
      status: {
        degraded: '本次未能取得近期資訊，因此只顯示地區背景與世界衛生組織兩層內容。',
        disabled: '連網搜尋目前為關閉狀態，因此只顯示地區背景與世界衛生組織兩層內容。',
      },
      toSymptoms: '身體不舒服？去做症狀自測',
    },
    travel: {
      label: '旅行背景',
      place: '目的地：{location}',
      level: '流行程度：{level}',
      notMatched: '參考表中沒有這個地點',
      offer: '要出門旅行嗎？',
      offerLink: '查詢目的地的登革熱情況',
    },
    privacy: {
      title: '隱私說明',
      intro: '這個工具在設計上就不需要知道你是誰。以下是它對你的資料所做的全部事情：',
      items: [
        '無需註冊、無需登入，不蒐集姓名、電話、電子郵件等任何聯絡方式。',
        '你的作答不會與任何身分資訊連結。',
        '「補充說明」中自由填寫的文字不會被保存。',
        '去識別化的作答與評分會被記錄，用於改進模型校準。',
        '不販售、也不與第三方共享你的資料。',
        '啟用 AI 建議時，你的作答摘要會傳送至第三方大型語言模型 API 進行處理。',
        '評估結果不是醫療診斷。',
      ],
      foot: '本說明描述本服務目前的實際運作方式；日後如有變動會同步更新。',
      close: '知道了',
    },
    brand: '🦟 登革熱風險自測',
    stepCounter: '第 {cur} 步 / 共 {total} 步',
    steps: {
      basic: { title: '基本資料' },
      common: { title: '常見症狀', },
      other: { title: '其他症狀', },
      clinical: { title: '出血與檢驗', sub: '根據醫師檢查結果' },
      exposure: { title: '周遭環境與暴露' },
      history: { title: '過去病史' },
      notes: { title: '補充說明', },
    },
    fields: {
      age: '年齡', ageUnit: '歲',
      sex: '性別', sexF: '女', sexM: '男',
      dayIll: '症狀已持續', dayIllUnit: '天',
      dayIllZero: '今天剛開始', dayIllNone: '尚無症狀',
    },
    answers: { yes: '有', no: '無', unknown: '不知道' },
    answered: '已答 {n}/{total}',
    notes: {
      placeholder: '例如：三天前開始發燒，眼睛後面很脹，昨天開始牙齦有點出血…',
    },
    nav: { prev: '上一步', next: '下一步', submit: '送出評估', back: '返回' },
    hints: { sexRequired: '請先選擇性別再繼續' },
    mode: {
      adaptive: '智慧問診（推薦）',
      classic: '完整問卷',
      switch: '選擇問診方式',
    },
    adaptive: {
      stageSafety: '安全篩檢',
      stageLoop: '智慧提問',
      questionCounter: '第 {n} 題',
      continueFull: '補充更多問題（轉完整問卷）',
    },
    loading: {
      steps: ['正在整理你的作答…', '模型評估中…', '正在產生個人化建議…'],
    },
    result: {
      title: '評估結果',
      dengue: '登革熱可能性',
      severe: '重症風險',
      worsening: '病情加重風險',
      levels: { low: '低風險', medium: '中風險', high: '高風險' },
      advice: { medical: '🏥 就醫提示', monitoring: '📋 居家監測', protection: '🛡️ 防蚊與防護' },
      explain: {
        why: '為什麼是這個分數？',
        title: '{metric} · 影響最大的因素',
        up: '推高評分',
        down: '拉低評分',
        empty: '這一項目前沒有可顯示的貢獻項。',
      },
      exposure: {
        label: '周遭暴露情況',
        levels: { low: '低', medium: '中', high: '高' },
        factors: '觸發因素：{list}',
        none: '你沒有回報周遭有相關的暴露情況。',
        sep: '、',
      },
      warning: {
        title: '出現登革熱警示徵象',
        body: '你回報了 {signs}，這屬於世界衛生組織列出的登革熱警示徵象。無論上方評分高低，都建議盡快就醫評估。',
        sep: '、',
      },
      restart: '重新評估',
      home: '返回首頁',
    },
    errors: {
      title: '評估失敗',
      network: '網路連線失敗，請檢查網路後重試。',
      validation: '送出的資訊有誤，請返回檢查後重試。',
      upstream: '模型服務暫時無法使用，請稍後重試。',
      server: '伺服器忙碌中，請稍後重試。',
      generic: '請求失敗（狀態碼 {status}），請稍後重試。',
      badData: '服務回傳的資料格式異常，請稍後重試。',
      retry: '重試',
      back: '返回修改',
    },
    disclaimer: '本結果僅供參考，不構成醫療診斷。如有不適請及時就醫。',
    symptoms: {
      FEBRE: { label: '發燒', desc: '體溫超過 37.5°C' },
      MIALGIA: { label: '肌肉痛', desc: '全身肌肉痠痛' },
      CEFALEIA: { label: '頭痛', desc: '持續性頭痛' },
      EXANTEMA: { label: '皮疹', desc: '皮膚出現紅色斑疹' },
      VOMITO: { label: '嘔吐', desc: '已經吐出胃內容物' },
      NAUSEA: { label: '噁心', desc: '想吐但沒有吐出來' },
      DOR_COSTAS: { label: '背痛', desc: '腰背部疼痛' },
      CONJUNTVIT: { label: '結膜炎', desc: '眼睛發紅、有分泌物' },
      ARTRITE: { label: '關節炎', desc: '關節紅腫發熱' },
      ARTRALGIA: { label: '關節痛', desc: '關節疼痛但沒有紅腫' },
      PETEQUIA_N: { label: '皮膚瘀點', desc: '針尖大小的紅點，按壓不褪色' },
      LEUCOPENIA: { label: '白血球減少', desc: '血液檢查顯示白血球偏低（需抽血）' },
      LACO: { label: '束臂試驗陽性', desc: '醫師用血壓計加壓後皮膚出現出血點' },
      DOR_RETRO: { label: '眼窩後疼痛', desc: '眼球後方疼痛，轉動眼球時加重' },
    },
    comorbidities: {
      DIABETES: { label: '糖尿病', desc: '已確診的糖尿病' },
      HEMATOLOG: { label: '血液疾病', desc: '如貧血、凝血障礙、血小板疾病' },
      HEPATOPAT: { label: '肝病', desc: '如肝炎、肝硬化' },
      RENAL: { label: '腎臟病', desc: '如慢性腎臟病、腎功能不全' },
      HIPERTENSA: { label: '高血壓', desc: '已確診的高血壓' },
      ACIDO_PEPT: { label: '消化性潰瘍', desc: '胃潰瘍或十二指腸潰瘍' },
      AUTO_IMUNE: { label: '自體免疫疾病', desc: '如紅斑性狼瘡、類風濕性關節炎' },
    },
    exposure: {
      FEVER_CLUSTER: { label: '周遭發燒病例增加', desc: '家人、同事、同學或鄰居近期發燒的人明顯變多' },
      CONFIRMED_CASE: { label: '身邊有確診病例', desc: '家庭、工作場所或社區裡有人被確診登革熱' },
      OUTBREAK_TRAVEL: { label: '疫情流行地區居住或旅遊', desc: '近期到訪過、或正居住於登革熱流行地區' },
    },
    features: {
      age: '年齡',
      sex_f: '女性',
      day_ill: '症狀持續天數',
      wk_sin: '季節性週期（分量一）',
      wk_cos: '季節性週期（分量二）',
    },
    chat: {
      title: '追問一下',
      inputLabel: '你的問題',
      placeholder: '輸入你的問題…',
      send: '送出',
      typing: 'AI 正在回覆…',
      you: '你',
      assistant: 'AI 助理',
      note: '回答由 AI 產生，僅供參考，不作為醫療診斷依據。',
      privacyLink: '隱私說明',
      error: 'AI 助理暫時無法回答，請稍後重試。',
      retry: '重試',
      chips: [
        '這些分數到底代表什麼？',
        '我現在需要去看醫師嗎？',
        '我該特別留意哪些危險徵兆？',
        '要怎麼保護家裡其他人？',
        '我很不舒服，為什麼重症風險還是低？',
        '這算是確診嗎？',
      ],
    },
  },

  en: {
    docTitle: 'Dengue Risk Self-Check',
    metaDescription: 'Answer questions about symptoms and medical history to get a dengue risk indicator and protection advice. For reference only, not a medical diagnosis.',
    langLabel: 'Select language',
    a11y: { progress: 'Questionnaire progress', chatLog: 'Conversation' },
    hero: {
      title: 'Dengue Risk<br />Self-Check',
      features: ['Dengue-specific', 'Built on real data', 'Five languages'],
      ctaTravelTitle: 'I’m travelling somewhere',
      ctaTravelDesc: 'See the dengue situation at that destination',
      ctaSymptomsTitle: 'I have symptoms',
      ctaSymptomsDesc: 'Answer a few questions for a risk indicator',
      privacyLink: 'Privacy',
    },
    sources: { label: 'Sources', who: 'World Health Organization', search: 'Web search', official: 'Official' },
    dest: {
      title: 'Destination check',
      sub: 'See what is known about dengue where you are going. This page gives no personal risk score.',
      back: 'Back',
      inputLabel: 'Destination',
      placeholder: 'Country, region or city',
      submit: 'Check',
      examplesLabel: 'For example',
      examples: ['Singapore', 'Brazil', 'Thailand', 'Philippines'],
      emptyHint: 'Please enter a destination first.',
      loading: {
        steps: ['Looking up regional background…', 'Searching recent reports…', 'Putting it together…'],
      },
      errors: {
        network: 'Network connection failed. Please check your connection and try again.',
        server: 'The destination service is temporarily unavailable. Please try again shortly.',
        generic: 'Request failed (status {status}). Please try again.',
        badData: 'The service returned unexpected data. Please try again.',
        retry: 'Try again',
      },
      resultTitle: 'Dengue in {location}',
      notMatched: '“{location}” could not be matched in our reference table, so no endemicity level is shown. Anything that was found is still listed below.',
      layers: {
        reference: {
          title: 'Regional background',
          tag: 'Stable reference',
        },
        who: {
          title: 'World Health Organization notices',
          tag: 'Official',
          empty: 'No World Health Organization notice was returned for this destination.',
        },
        recent: {
          title: 'Recent reports',
          tag: 'Web search · last ~3 months',
          empty: 'No recent reports were found.',
        },
      },
      endemicity: {
        label: 'Endemicity',
        levels: { high: 'High', moderate: 'Moderate', low: 'Low', none: 'No local transmission reported', unknown: 'Unknown' },
      },
      seasonLabel: 'Season',
      status: {
        degraded: 'Recent information could not be retrieved this time, so only the regional background and World Health Organization layers are shown.',
        disabled: 'Live web search is currently switched off, so only the regional background and World Health Organization layers are shown.',
      },
      toSymptoms: 'Feeling unwell? Check your symptoms',
    },
    travel: {
      label: 'Travel context',
      place: 'Destination: {location}',
      level: 'Endemicity: {level}',
      notMatched: 'Not in the reference table',
      offer: 'Travelling somewhere?',
      offerLink: 'Check the dengue situation there',
    },
    privacy: {
      title: 'Privacy notice',
      intro: 'This tool is designed to work without knowing who you are. Here is everything it does with your data:',
      items: [
        'No account, no login. We do not collect your name, phone number, email or any other contact details.',
        'Your answers are not linked to any identity.',
        'Free-text notes are never stored.',
        'De-identified answers and scores are logged so the model can be better calibrated.',
        'Nothing is sold or shared with third parties.',
        'When AI advice is enabled, a summary of your answers is sent to a third-party language-model API for processing.',
        'Results are not a medical diagnosis.',
      ],
      foot: 'This notice describes how the service actually works today; it will be updated if that changes.',
      close: 'Got it',
    },
    brand: '🦟 Dengue Risk Self-Check',
    stepCounter: 'Step {cur} of {total}',
    steps: {
      basic: { title: 'Basic information' },
      common: { title: 'Common symptoms', },
      other: { title: 'Other symptoms', },
      clinical: { title: 'Bleeding & lab findings', sub: 'Based on your doctor’s findings' },
      exposure: { title: 'Surroundings & exposure' },
      history: { title: 'Medical history' },
      notes: { title: 'Anything else', },
    },
    fields: {
      age: 'Age', ageUnit: 'years',
      sex: 'Sex', sexF: 'Female', sexM: 'Male',
      dayIll: 'Symptoms have lasted', dayIllUnit: 'days',
      dayIllZero: 'Started today', dayIllNone: 'No symptoms yet',
    },
    answers: { yes: 'Yes', no: 'No', unknown: 'Don’t know' },
    answered: 'Answered {n}/{total}',
    notes: {
      placeholder: 'e.g. Fever started three days ago, pressure behind the eyes, gums began bleeding yesterday…',
    },
    nav: { prev: 'Back', next: 'Next', submit: 'Get my result', back: 'Back' },
    hints: { sexRequired: 'Please select your sex to continue' },
    mode: {
      adaptive: 'Smart interview (recommended)',
      classic: 'Full questionnaire',
      switch: 'Choose how questions are asked',
    },
    adaptive: {
      stageSafety: 'Safety check',
      stageLoop: 'Smart questions',
      questionCounter: 'Question {n}',
      continueFull: 'Answer more questions (full questionnaire)',
    },
    loading: {
      steps: ['Organising your answers…', 'Running the model…', 'Preparing your advice…'],
    },
    result: {
      title: 'Your result',
      dengue: 'Dengue likelihood',
      severe: 'Severe-disease risk',
      worsening: 'Risk of worsening',
      levels: { low: 'Low risk', medium: 'Moderate risk', high: 'High risk' },
      advice: { medical: '🏥 When to seek care', monitoring: '📋 Monitoring at home', protection: '🛡️ Mosquito protection' },
      explain: {
        why: 'Why this score?',
        title: '{metric} · largest contributors',
        up: 'raises the score',
        down: 'lowers the score',
        empty: 'No contributors are available for this score.',
      },
      exposure: {
        label: 'Surrounding exposure',
        levels: { low: 'Low', medium: 'Medium', high: 'High' },
        factors: 'Triggered by: {list}',
        none: 'You did not report any relevant exposure around you.',
        sep: ', ',
      },
      warning: {
        title: 'Dengue warning sign present',
        body: 'You reported {signs}, which the World Health Organization lists as a dengue warning sign. Regardless of the scores above, you should seek medical assessment promptly.',
        sep: ', ',
      },
      restart: 'Start over',
      home: 'Back to home',
    },
    errors: {
      title: 'Assessment failed',
      network: 'Network connection failed. Please check your connection and try again.',
      validation: 'Some of the submitted information was invalid. Please go back and check.',
      upstream: 'The model service is temporarily unavailable. Please try again shortly.',
      server: 'Something went wrong on our side. Please try again shortly.',
      generic: 'Request failed (status {status}). Please try again.',
      badData: 'The service returned unexpected data. Please try again.',
      retry: 'Try again',
      back: 'Go back and edit',
    },
    disclaimer: 'This result is for reference only and does not constitute a medical diagnosis. Please seek medical care if you feel unwell.',
    symptoms: {
      FEBRE: { label: 'Fever', desc: 'Temperature above 37.5°C' },
      MIALGIA: { label: 'Muscle pain', desc: 'Aching muscles throughout the body' },
      CEFALEIA: { label: 'Headache', desc: 'Persistent headache' },
      EXANTEMA: { label: 'Rash', desc: 'Red blotchy rash on the skin' },
      VOMITO: { label: 'Vomiting', desc: 'You have actually vomited' },
      NAUSEA: { label: 'Nausea', desc: 'Feeling sick without vomiting' },
      DOR_COSTAS: { label: 'Back pain', desc: 'Pain in the lower or upper back' },
      CONJUNTVIT: { label: 'Conjunctivitis', desc: 'Red eyes with discharge' },
      ARTRITE: { label: 'Arthritis', desc: 'Joints swollen, red and warm' },
      ARTRALGIA: { label: 'Joint pain', desc: 'Painful joints without swelling' },
      PETEQUIA_N: { label: 'Petechiae', desc: 'Pinpoint red spots that do not fade when pressed' },
      LEUCOPENIA: { label: 'Low white blood cell count', desc: 'Shown by a blood test' },
      LACO: { label: 'Positive tourniquet test', desc: 'Bleeding spots appear after a doctor inflates a blood-pressure cuff' },
      DOR_RETRO: { label: 'Pain behind the eyes', desc: 'Worse when moving the eyes' },
    },
    comorbidities: {
      DIABETES: { label: 'Diabetes', desc: 'Diagnosed diabetes' },
      HEMATOLOG: { label: 'Blood disorder', desc: 'e.g. anaemia, clotting or platelet disorders' },
      HEPATOPAT: { label: 'Liver disease', desc: 'e.g. hepatitis, cirrhosis' },
      RENAL: { label: 'Kidney disease', desc: 'e.g. chronic kidney disease, renal insufficiency' },
      HIPERTENSA: { label: 'Hypertension', desc: 'Diagnosed high blood pressure' },
      ACIDO_PEPT: { label: 'Peptic ulcer disease', desc: 'Stomach or duodenal ulcer' },
      AUTO_IMUNE: { label: 'Autoimmune disease', desc: 'e.g. lupus, rheumatoid arthritis' },
    },
    exposure: {
      FEVER_CLUSTER: { label: 'Cluster of fever cases nearby', desc: 'An unusual increase in fever cases among the people around you — family, school, workplace or neighbourhood' },
      CONFIRMED_CASE: { label: 'Confirmed dengue case nearby', desc: 'Someone in your household, workplace or neighbourhood has been diagnosed with dengue' },
      OUTBREAK_TRAVEL: { label: 'Outbreak area', desc: 'You live in, or have recently travelled to, an area with a dengue outbreak' },
    },
    features: {
      age: 'Age',
      sex_f: 'Female sex',
      day_ill: 'Days since symptoms began',
      wk_sin: 'Seasonal pattern (component 1)',
      wk_cos: 'Seasonal pattern (component 2)',
    },
    chat: {
      title: 'Ask about your result',
      inputLabel: 'Your question',
      placeholder: 'Type your question…',
      send: 'Send',
      typing: 'Assistant is replying…',
      you: 'You',
      assistant: 'Assistant',
      note: 'Replies are AI-generated, for reference only, and not a basis for medical diagnosis.',
      privacyLink: 'Privacy',
      error: 'The assistant could not answer just now. Please try again.',
      retry: 'Retry',
      chips: [
        'What do these scores actually mean?',
        'Should I see a doctor now?',
        'What warning signs should I watch for?',
        'How do I protect my family?',
        'Why is my severity score low when I feel awful?',
        'Is this a diagnosis?',
      ],
    },
  },

  es: {
    docTitle: 'Autoevaluación de riesgo de dengue',
    metaDescription: 'Responda preguntas sobre síntomas y antecedentes para obtener un indicador de riesgo de dengue y recomendaciones de protección. Solo orientativo, no es un diagnóstico médico.',
    langLabel: 'Seleccionar idioma',
    a11y: { progress: 'Progreso del cuestionario', chatLog: 'Conversación' },
    hero: {
      title: 'Riesgo de dengue<br />Autoevaluación',
      features: ['Específico para dengue', 'Basado en datos reales', 'Cinco idiomas'],
      ctaTravelTitle: 'Voy a viajar',
      ctaTravelDesc: 'Consulte la situación del dengue en ese destino',
      ctaSymptomsTitle: 'Tengo síntomas',
      ctaSymptomsDesc: 'Responda unas preguntas y obtenga un indicador de riesgo',
      privacyLink: 'Privacidad',
    },
    sources: { label: 'Fuentes', who: 'Organización Mundial de la Salud', search: 'Búsqueda web', official: 'Oficial' },
    dest: {
      title: 'Consulta de destino',
      sub: 'Vea lo que se sabe sobre el dengue en el lugar al que va. Esta página no ofrece ninguna puntuación de riesgo personal.',
      back: 'Volver',
      inputLabel: 'Destino',
      placeholder: 'País, región o ciudad',
      submit: 'Consultar',
      examplesLabel: 'Por ejemplo',
      examples: ['Singapur', 'Brasil', 'Tailandia', 'Filipinas'],
      emptyHint: 'Escriba primero un destino.',
      loading: {
        steps: ['Consultando el contexto regional…', 'Buscando informes recientes…', 'Reuniendo la información…'],
      },
      errors: {
        network: 'Fallo de conexión. Revise su red e inténtelo de nuevo.',
        server: 'El servicio de consulta de destinos no está disponible temporalmente. Inténtelo más tarde.',
        generic: 'La solicitud falló (código {status}). Inténtelo de nuevo.',
        badData: 'El servicio devolvió datos inesperados. Inténtelo de nuevo.',
        retry: 'Reintentar',
      },
      resultTitle: 'Dengue en {location}',
      notMatched: 'No se pudo localizar «{location}» en nuestra tabla de referencia, por lo que no se muestra un nivel de endemicidad. Lo que sí se encontró se indica igualmente más abajo.',
      layers: {
        reference: {
          title: 'Contexto regional',
          tag: 'Referencia estable',
        },
        who: {
          title: 'Avisos de la Organización Mundial de la Salud',
          tag: 'Oficial',
          empty: 'No se obtuvo ningún aviso de la Organización Mundial de la Salud para este destino.',
        },
        recent: {
          title: 'Informes recientes',
          tag: 'Búsqueda web · últimos ~3 meses',
          empty: 'No se encontraron informes recientes.',
        },
      },
      endemicity: {
        label: 'Endemicidad',
        levels: { high: 'Alta', moderate: 'Moderada', low: 'Baja', none: 'Sin transmisión local notificada', unknown: 'Desconocida' },
      },
      seasonLabel: 'Temporada',
      status: {
        degraded: 'Esta vez no se ha podido obtener información reciente, así que solo se muestran las capas de contexto regional y de la Organización Mundial de la Salud.',
        disabled: 'La búsqueda web en vivo está desactivada, así que solo se muestran las capas de contexto regional y de la Organización Mundial de la Salud.',
      },
      toSymptoms: '¿No se encuentra bien? Evalúe sus síntomas',
    },
    travel: {
      label: 'Contexto de viaje',
      place: 'Destino: {location}',
      level: 'Endemicidad: {level}',
      notMatched: 'No figura en la tabla de referencia',
      offer: '¿Va a viajar?',
      offerLink: 'Consulte la situación del dengue allí',
    },
    privacy: {
      title: 'Aviso de privacidad',
      intro: 'Esta herramienta está diseñada para funcionar sin saber quién es usted. Esto es todo lo que hace con sus datos:',
      items: [
        'Sin cuenta ni inicio de sesión. No recogemos su nombre, teléfono, correo electrónico ni ningún otro dato de contacto.',
        'Sus respuestas no se vinculan a ninguna identidad.',
        'El texto libre de los comentarios nunca se almacena.',
        'Las respuestas y puntuaciones anonimizadas se registran para mejorar la calibración del modelo.',
        'No vendemos ni compartimos sus datos con terceros.',
        'Cuando las recomendaciones con IA están activadas, se envía un resumen de sus respuestas a una API de modelo de lenguaje de terceros para su procesamiento.',
        'Los resultados no son un diagnóstico médico.',
      ],
      foot: 'Este aviso describe cómo funciona el servicio en la actualidad; se actualizará si eso cambia.',
      close: 'Entendido',
    },
    brand: '🦟 Autoevaluación de dengue',
    stepCounter: 'Paso {cur} de {total}',
    steps: {
      basic: { title: 'Datos básicos' },
      common: { title: 'Síntomas frecuentes', },
      other: { title: 'Otros síntomas', },
      clinical: { title: 'Sangrado y laboratorio', sub: 'Según los hallazgos de su médico' },
      exposure: { title: 'Entorno y exposición' },
      history: { title: 'Antecedentes médicos' },
      notes: { title: 'Comentarios', },
    },
    fields: {
      age: 'Edad', ageUnit: 'años',
      sex: 'Sexo', sexF: 'Mujer', sexM: 'Hombre',
      dayIll: 'Los síntomas llevan', dayIllUnit: 'días',
      dayIllZero: 'Comenzaron hoy', dayIllNone: 'Aún sin síntomas',
    },
    answers: { yes: 'Sí', no: 'No', unknown: 'No sé' },
    answered: 'Respondidas {n}/{total}',
    notes: {
      placeholder: 'Ej.: la fiebre empezó hace tres días, presión detrás de los ojos, ayer comenzó a sangrar la encía…',
    },
    nav: { prev: 'Atrás', next: 'Siguiente', submit: 'Ver mi resultado', back: 'Volver' },
    hints: { sexRequired: 'Seleccione su sexo para continuar' },
    mode: {
      adaptive: 'Entrevista inteligente (recomendada)',
      classic: 'Cuestionario completo',
      switch: 'Elegir el modo de preguntas',
    },
    adaptive: {
      stageSafety: 'Preguntas de seguridad',
      stageLoop: 'Preguntas inteligentes',
      questionCounter: 'Pregunta {n}',
      continueFull: 'Responder más preguntas (cuestionario completo)',
    },
    loading: {
      steps: ['Organizando sus respuestas…', 'Ejecutando el modelo…', 'Preparando sus recomendaciones…'],
    },
    result: {
      title: 'Su resultado',
      dengue: 'Probabilidad relativa de dengue',
      severe: 'Riesgo de gravedad',
      worsening: 'Riesgo de empeoramiento',
      levels: { low: 'Riesgo bajo', medium: 'Riesgo moderado', high: 'Riesgo alto' },
      advice: { medical: '🏥 Cuándo buscar atención', monitoring: '📋 Vigilancia en casa', protection: '🛡️ Protección contra mosquitos' },
      explain: {
        why: '¿Por qué esta puntuación?',
        title: '{metric} · factores más influyentes',
        up: 'aumenta la puntuación',
        down: 'reduce la puntuación',
        empty: 'No hay factores disponibles para esta puntuación.',
      },
      exposure: {
        label: 'Exposición en su entorno',
        levels: { low: 'Baja', medium: 'Media', high: 'Alta' },
        factors: 'Motivos: {list}',
        none: 'No ha indicado ninguna exposición relevante en su entorno.',
        sep: ', ',
      },
      warning: {
        title: 'Signo de alarma de dengue presente',
        body: 'Ha indicado {signs}, que la Organización Mundial de la Salud considera un signo de alarma del dengue. Independientemente de las puntuaciones anteriores, conviene buscar valoración médica cuanto antes.',
        sep: ', ',
      },
      restart: 'Empezar de nuevo',
      home: 'Volver al inicio',
    },
    errors: {
      title: 'Error en la evaluación',
      network: 'Fallo de conexión. Revise su red e inténtelo de nuevo.',
      validation: 'Algunos datos enviados no son válidos. Vuelva atrás y revíselos.',
      upstream: 'El servicio del modelo no está disponible temporalmente. Inténtelo más tarde.',
      server: 'Se produjo un error en el servidor. Inténtelo más tarde.',
      generic: 'La solicitud falló (código {status}). Inténtelo de nuevo.',
      badData: 'El servicio devolvió datos inesperados. Inténtelo de nuevo.',
      retry: 'Reintentar',
      back: 'Volver y editar',
    },
    disclaimer: 'Este resultado es solo orientativo y no constituye un diagnóstico médico. Si se siente mal, busque atención médica.',
    symptoms: {
      FEBRE: { label: 'Fiebre', desc: 'Temperatura superior a 37,5 °C' },
      MIALGIA: { label: 'Mialgia', desc: 'Dolor muscular generalizado' },
      CEFALEIA: { label: 'Cefalea', desc: 'Dolor de cabeza persistente' },
      EXANTEMA: { label: 'Erupción cutánea', desc: 'Manchas rojas en la piel' },
      VOMITO: { label: 'Vómitos', desc: 'Ha llegado a vomitar' },
      NAUSEA: { label: 'Náuseas', desc: 'Ganas de vomitar sin llegar a hacerlo' },
      DOR_COSTAS: { label: 'Dolor de espalda', desc: 'Dolor lumbar o dorsal' },
      CONJUNTVIT: { label: 'Conjuntivitis', desc: 'Ojos rojos con secreción' },
      ARTRITE: { label: 'Artritis', desc: 'Articulaciones hinchadas, rojas y calientes' },
      ARTRALGIA: { label: 'Artralgia', desc: 'Dolor articular sin hinchazón' },
      PETEQUIA_N: { label: 'Petequias', desc: 'Puntitos rojos que no desaparecen al presionar' },
      LEUCOPENIA: { label: 'Leucopenia', desc: 'Recuento bajo de glóbulos blancos en el análisis' },
      LACO: { label: 'Prueba del torniquete positiva', desc: 'Aparecen petequias tras inflar el manguito de presión' },
      DOR_RETRO: { label: 'Dolor retroocular', desc: 'Dolor detrás de los ojos, peor al moverlos' },
    },
    comorbidities: {
      DIABETES: { label: 'Diabetes', desc: 'Diabetes diagnosticada' },
      HEMATOLOG: { label: 'Enfermedad hematológica', desc: 'Ej.: anemia, trastornos de coagulación o plaquetas' },
      HEPATOPAT: { label: 'Hepatopatía', desc: 'Ej.: hepatitis, cirrosis' },
      RENAL: { label: 'Enfermedad renal', desc: 'Ej.: enfermedad renal crónica, insuficiencia renal' },
      HIPERTENSA: { label: 'Hipertensión', desc: 'Presión arterial alta diagnosticada' },
      ACIDO_PEPT: { label: 'Enfermedad ácido-péptica', desc: 'Úlcera gástrica o duodenal' },
      AUTO_IMUNE: { label: 'Enfermedad autoinmune', desc: 'Ej.: lupus, artritis reumatoide' },
    },
    exposure: {
      FEVER_CLUSTER: { label: 'Aumento de casos de fiebre cerca', desc: 'Un aumento inusual de casos de fiebre entre las personas de su entorno: familia, escuela, trabajo o vecindario' },
      CONFIRMED_CASE: { label: 'Caso confirmado de dengue cerca', desc: 'Alguien de su hogar, trabajo o vecindario ha sido diagnosticado de dengue' },
      OUTBREAK_TRAVEL: { label: 'Zona con brote', desc: 'Vive en una zona con brote de dengue o ha viajado a una recientemente' },
    },
    features: {
      age: 'Edad',
      sex_f: 'Sexo femenino',
      day_ill: 'Días desde el inicio de los síntomas',
      wk_sin: 'Patrón estacional (componente 1)',
      wk_cos: 'Patrón estacional (componente 2)',
    },
    chat: {
      title: 'Pregunte sobre su resultado',
      inputLabel: 'Su pregunta',
      placeholder: 'Escriba su pregunta…',
      send: 'Enviar',
      typing: 'El asistente está respondiendo…',
      you: 'Usted',
      assistant: 'Asistente',
      note: 'Las respuestas las genera una IA, son solo orientativas y no sirven como base para un diagnóstico médico.',
      privacyLink: 'Privacidad',
      error: 'El asistente no ha podido responder ahora mismo. Inténtelo de nuevo.',
      retry: 'Reintentar',
      chips: [
        '¿Qué significan realmente estas puntuaciones?',
        '¿Debo ir al médico ahora?',
        '¿Qué signos de alarma debo vigilar?',
        '¿Cómo protejo a mi familia?',
        'Me siento fatal, ¿por qué mi riesgo de gravedad es bajo?',
        '¿Esto es un diagnóstico?',
      ],
    },
  },

  pt: {
    docTitle: 'Autoavaliação de risco de dengue',
    metaDescription: 'Responda a perguntas sobre sintomas e histórico para obter um indicador de risco de dengue e orientações de proteção. Apenas para referência, não é um diagnóstico médico.',
    langLabel: 'Selecionar idioma',
    a11y: { progress: 'Progresso do questionário', chatLog: 'Conversa' },
    hero: {
      title: 'Risco de dengue<br />Autoavaliação',
      features: ['Específico para dengue', 'Baseado em dados reais', 'Cinco idiomas'],
      ctaTravelTitle: 'Vou viajar',
      ctaTravelDesc: 'Veja a situação da dengue nesse destino',
      ctaSymptomsTitle: 'Tenho sintomas',
      ctaSymptomsDesc: 'Responda a algumas perguntas e receba um indicador de risco',
      privacyLink: 'Privacidade',
    },
    sources: { label: 'Fontes', who: 'Organização Mundial da Saúde', search: 'Busca na web', official: 'Oficial' },
    dest: {
      title: 'Consulta de destino',
      sub: 'Veja o que se sabe sobre a dengue no lugar para onde você vai. Esta página não fornece nenhuma pontuação de risco pessoal.',
      back: 'Voltar',
      inputLabel: 'Destino',
      placeholder: 'País, região ou cidade',
      submit: 'Consultar',
      examplesLabel: 'Por exemplo',
      examples: ['Singapura', 'Brasil', 'Tailândia', 'Filipinas'],
      emptyHint: 'Digite primeiro um destino.',
      loading: {
        steps: ['Consultando o contexto regional…', 'Buscando relatos recentes…', 'Reunindo as informações…'],
      },
      errors: {
        network: 'Falha de conexão. Verifique sua rede e tente novamente.',
        server: 'O serviço de consulta de destinos está temporariamente indisponível. Tente novamente em instantes.',
        generic: 'A solicitação falhou (código {status}). Tente novamente.',
        badData: 'O serviço retornou dados inesperados. Tente novamente.',
        retry: 'Tentar novamente',
      },
      resultTitle: 'Dengue em {location}',
      notMatched: 'Não foi possível localizar «{location}» na nossa tabela de referência, por isso nenhum nível de endemicidade é exibido. O que foi encontrado continua listado abaixo.',
      layers: {
        reference: {
          title: 'Contexto regional',
          tag: 'Referência estável',
        },
        who: {
          title: 'Comunicados da Organização Mundial da Saúde',
          tag: 'Oficial',
          empty: 'Nenhum comunicado da Organização Mundial da Saúde foi retornado para este destino.',
        },
        recent: {
          title: 'Relatos recentes',
          tag: 'Busca na web · últimos ~3 meses',
          empty: 'Nenhum relato recente foi encontrado.',
        },
      },
      endemicity: {
        label: 'Endemicidade',
        levels: { high: 'Alta', moderate: 'Moderada', low: 'Baixa', none: 'Sem transmissão local notificada', unknown: 'Desconhecida' },
      },
      seasonLabel: 'Temporada',
      status: {
        degraded: 'Desta vez não foi possível obter informações recentes, portanto apenas as camadas de contexto regional e da Organização Mundial da Saúde são exibidas.',
        disabled: 'A busca na web ao vivo está desativada no momento, portanto apenas as camadas de contexto regional e da Organização Mundial da Saúde são exibidas.',
      },
      toSymptoms: 'Não está se sentindo bem? Avalie seus sintomas',
    },
    travel: {
      label: 'Contexto de viagem',
      place: 'Destino: {location}',
      level: 'Endemicidade: {level}',
      notMatched: 'Não consta na tabela de referência',
      offer: 'Vai viajar?',
      offerLink: 'Consulte a situação da dengue lá',
    },
    privacy: {
      title: 'Aviso de privacidade',
      intro: 'Esta ferramenta foi feita para funcionar sem saber quem você é. Isto é tudo o que ela faz com os seus dados:',
      items: [
        'Sem cadastro nem login. Não coletamos nome, telefone, e-mail nem qualquer outro dado de contato.',
        'Suas respostas não são vinculadas a nenhuma identidade.',
        'O texto livre das observações nunca é armazenado.',
        'Respostas e pontuações não identificadas são registradas para melhorar a calibração do modelo.',
        'Nada é vendido nem compartilhado com terceiros.',
        'Quando as orientações por IA estão ativadas, um resumo das suas respostas é enviado a uma API de modelo de linguagem de terceiros para processamento.',
        'Os resultados não são um diagnóstico médico.',
      ],
      foot: 'Este aviso descreve como o serviço funciona hoje; será atualizado caso isso mude.',
      close: 'Entendi',
    },
    brand: '🦟 Autoavaliação de dengue',
    stepCounter: 'Etapa {cur} de {total}',
    steps: {
      basic: { title: 'Dados básicos' },
      common: { title: 'Sintomas comuns', },
      other: { title: 'Outros sintomas', },
      clinical: { title: 'Sangramento e exames', sub: 'Conforme os achados do seu médico' },
      exposure: { title: 'Ambiente e exposição' },
      history: { title: 'Histórico médico' },
      notes: { title: 'Observações', },
    },
    fields: {
      age: 'Idade', ageUnit: 'anos',
      sex: 'Sexo', sexF: 'Feminino', sexM: 'Masculino',
      dayIll: 'Os sintomas duram há', dayIllUnit: 'dias',
      dayIllZero: 'Começaram hoje', dayIllNone: 'Ainda sem sintomas',
    },
    answers: { yes: 'Sim', no: 'Não', unknown: 'Não sei' },
    answered: 'Respondidas {n}/{total}',
    notes: {
      placeholder: 'Ex.: a febre começou há três dias, pressão atrás dos olhos, ontem a gengiva começou a sangrar…',
    },
    nav: { prev: 'Voltar', next: 'Avançar', submit: 'Ver meu resultado', back: 'Voltar' },
    hints: { sexRequired: 'Selecione o sexo para continuar' },
    mode: {
      adaptive: 'Entrevista inteligente (recomendada)',
      classic: 'Questionário completo',
      switch: 'Escolher o modo das perguntas',
    },
    adaptive: {
      stageSafety: 'Perguntas de segurança',
      stageLoop: 'Perguntas inteligentes',
      questionCounter: 'Pergunta {n}',
      continueFull: 'Responder mais perguntas (questionário completo)',
    },
    loading: {
      steps: ['Organizando suas respostas…', 'Executando o modelo…', 'Preparando suas orientações…'],
    },
    result: {
      title: 'Seu resultado',
      dengue: 'Probabilidade relativa de dengue',
      severe: 'Risco de gravidade',
      worsening: 'Risco de piora',
      levels: { low: 'Risco baixo', medium: 'Risco moderado', high: 'Risco alto' },
      advice: { medical: '🏥 Quando procurar atendimento', monitoring: '📋 Monitoramento em casa', protection: '🛡️ Proteção contra mosquitos' },
      explain: {
        why: 'Por que esta pontuação?',
        title: '{metric} · fatores de maior peso',
        up: 'aumenta a pontuação',
        down: 'reduz a pontuação',
        empty: 'Não há fatores disponíveis para esta pontuação.',
      },
      exposure: {
        label: 'Exposição ao seu redor',
        levels: { low: 'Baixa', medium: 'Média', high: 'Alta' },
        factors: 'Motivos: {list}',
        none: 'Você não relatou nenhuma exposição relevante ao seu redor.',
        sep: ', ',
      },
      warning: {
        title: 'Sinal de alarme de dengue presente',
        body: 'Você relatou {signs}, que a Organização Mundial da Saúde considera um sinal de alarme da dengue. Independentemente das pontuações acima, procure avaliação médica o quanto antes.',
        sep: ', ',
      },
      restart: 'Recomeçar',
      home: 'Voltar ao início',
    },
    errors: {
      title: 'Falha na avaliação',
      network: 'Falha de conexão. Verifique sua rede e tente novamente.',
      validation: 'Alguns dados enviados são inválidos. Volte e verifique.',
      upstream: 'O serviço do modelo está temporariamente indisponível. Tente novamente em instantes.',
      server: 'Ocorreu um erro no servidor. Tente novamente em instantes.',
      generic: 'A solicitação falhou (código {status}). Tente novamente.',
      badData: 'O serviço retornou dados inesperados. Tente novamente.',
      retry: 'Tentar novamente',
      back: 'Voltar e editar',
    },
    disclaimer: 'Este resultado é apenas para referência e não constitui um diagnóstico médico. Se não se sentir bem, procure atendimento médico.',
    symptoms: {
      FEBRE: { label: 'Febre', desc: 'Temperatura acima de 37,5 °C' },
      MIALGIA: { label: 'Mialgia', desc: 'Dor muscular pelo corpo' },
      CEFALEIA: { label: 'Cefaleia', desc: 'Dor de cabeça persistente' },
      EXANTEMA: { label: 'Exantema', desc: 'Manchas vermelhas na pele' },
      VOMITO: { label: 'Vômito', desc: 'Chegou a vomitar' },
      NAUSEA: { label: 'Náusea', desc: 'Enjoo sem chegar a vomitar' },
      DOR_COSTAS: { label: 'Dor nas costas', desc: 'Dor lombar ou dorsal' },
      CONJUNTVIT: { label: 'Conjuntivite', desc: 'Olhos vermelhos com secreção' },
      ARTRITE: { label: 'Artrite', desc: 'Articulações inchadas, vermelhas e quentes' },
      ARTRALGIA: { label: 'Artralgia', desc: 'Dor nas articulações sem inchaço' },
      PETEQUIA_N: { label: 'Petéquias', desc: 'Pontinhos vermelhos que não somem ao pressionar' },
      LEUCOPENIA: { label: 'Leucopenia', desc: 'Exame de sangue mostrando leucócitos baixos' },
      LACO: { label: 'Prova do laço positiva', desc: 'Surgem petéquias após o médico inflar o manguito de pressão' },
      DOR_RETRO: { label: 'Dor retro-orbital', desc: 'Dor atrás dos olhos, piora ao movê-los' },
    },
    comorbidities: {
      DIABETES: { label: 'Diabetes', desc: 'Diabetes diagnosticada' },
      HEMATOLOG: { label: 'Doença hematológica', desc: 'Ex.: anemia, distúrbios de coagulação ou plaquetas' },
      HEPATOPAT: { label: 'Hepatopatia', desc: 'Ex.: hepatite, cirrose' },
      RENAL: { label: 'Doença renal', desc: 'Ex.: doença renal crônica, insuficiência renal' },
      HIPERTENSA: { label: 'Hipertensão', desc: 'Pressão alta diagnosticada' },
      ACIDO_PEPT: { label: 'Doença ácido-péptica', desc: 'Úlcera gástrica ou duodenal' },
      AUTO_IMUNE: { label: 'Doença autoimune', desc: 'Ex.: lúpus, artrite reumatoide' },
    },
    exposure: {
      FEVER_CLUSTER: { label: 'Aumento de casos de febre por perto', desc: 'Aumento incomum de casos de febre entre as pessoas ao seu redor: família, escola, trabalho ou vizinhança' },
      CONFIRMED_CASE: { label: 'Caso confirmado de dengue por perto', desc: 'Alguém da sua casa, do trabalho ou da vizinhança foi diagnosticado com dengue' },
      OUTBREAK_TRAVEL: { label: 'Área com surto', desc: 'Você mora em uma área com surto de dengue ou viajou para uma recentemente' },
    },
    features: {
      age: 'Idade',
      sex_f: 'Sexo feminino',
      day_ill: 'Dias desde o início dos sintomas',
      wk_sin: 'Padrão sazonal (componente 1)',
      wk_cos: 'Padrão sazonal (componente 2)',
    },
    chat: {
      title: 'Pergunte sobre o seu resultado',
      inputLabel: 'Sua pergunta',
      placeholder: 'Escreva sua pergunta…',
      send: 'Enviar',
      typing: 'O assistente está respondendo…',
      you: 'Você',
      assistant: 'Assistente',
      note: 'As respostas são geradas por IA, servem apenas como referência e não constituem base para diagnóstico médico.',
      privacyLink: 'Privacidade',
      error: 'O assistente não conseguiu responder agora. Tente novamente.',
      retry: 'Tentar novamente',
      chips: [
        'O que essas pontuações realmente significam?',
        'Devo procurar um médico agora?',
        'Quais sinais de alarme devo observar?',
        'Como protejo minha família?',
        'Me sinto péssimo, por que meu risco de gravidade está baixo?',
        'Isto é um diagnóstico?',
      ],
    },
  },
};

// The two privacy items that need bolding (the order is the same in every language):
// free text is not stored, and the AI sends a summary off-site
const PRIVACY_KEY_ITEMS = [2, 5];

/* --------------------------------------------------------- State */

function freshAnswers() {
  const symptoms = {};
  SYMPTOM_CODES.forEach((c) => { symptoms[c] = 'unknown'; });
  const comorbidities = {};
  COMORB_CODES.forEach((c) => { comorbidities[c] = 'unknown'; });
  const exposure = {};
  EXPOSURE_CODES.forEach((c) => { exposure[c] = 'unknown'; });
  return { symptoms, comorbidities, exposure, age: 30, sex: null, dayIll: 2, notes: '' };
}

function freshChat() {
  return { messages: [], sending: false };
}

/**
 * State of the destination lookup. It survives across views (after finishing the
 * symptom self-assessment, the result page needs to be able to refer to a destination
 * looked up during this session), so resetWizard does not clear it.
 */
function freshDestination() {
  return {
    query: '',      // what is in the input box
    location: '',   // the most recently submitted location (for retries)
    data: null,     // the /api/destination response
    loading: false,
    error: null,    // { kind } | { status, detail }
    hint: false,    // whether the empty-input hint is visible (must follow language changes)
    seq: 0,         // request sequence number, for discarding stale responses
    from: 'hero',   // which view we came in from, deciding where "back" goes
  };
}

/**
 * The "answered" markers, kept separate from the values in answers:
 * every three-state question in answers defaults to 'unknown', which cannot
 * distinguish "the user explicitly chose don't know" from "never asked at all".
 * The /api/plan contract is: a key present = answered (yes/no/unknown all count),
 * a key absent = not asked yet -- so this has to be recorded separately.
 */
function freshAnsweredSet() {
  return { symptoms: {}, comorbidities: {}, exposure: {} };
}

function freshAdaptive() {
  return {
    stage: 'basic',    // basic → safety → exposure → loop → notes
    current: null,     // the current question { kind, code, why }
    loopCount: 0,      // questions answered in the adaptive stage (excluding safety/exposure)
    stopReason: null,  // 'proven' | 'cap' | 'nomore'
    planning: false,   // an /api/plan request is in flight
    planSeq: 0,        // request sequence number, for discarding stale responses
  };
}

const state = {
  lang: DEFAULT_LANG,
  mode: 'adaptive', // 'adaptive' | 'classic'; init reads this from localStorage
  step: 0,
  // Which view the assessment was entered from, deciding where its back control goes
  wizardFrom: 'hero',
  answers: freshAnswers(),
  answered: freshAnsweredSet(),
  adaptive: freshAdaptive(),
  submitting: false,
  // Expanded state of the three contribution panels (must survive a language change)
  openExplain: { dengue: false, worsening: false, severe: false },
  chat: freshChat(),
  destination: freshDestination(),
};

let lastResult = null;
let lastError = null;
let loadingTimer = null;
let destLoadingTimer = null;
let lastFocusBeforeModal = null;

/* --------------------------------------------------------- Utilities */

const $ = (id) => document.getElementById(id);
const T = () => I18N[state.lang];
const fmt = (tpl, vars) => tpl.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));

function itemText(kind, code) {
  const bag = T()[kind];
  return (bag && bag[code]) || { label: code, desc: '' };
}

/**
 * Turn a code from the back end's explanations into human-readable text.
 * Symptoms / comorbidities are looked up after dropping the _x suffix; the 5
 * non-binary features are looked up in the features table.
 */
function featureLabel(code) {
  const t = T();
  const raw = String(code || '');
  const base = raw.replace(/_x$/, '');
  if (NON_BINARY_FEATS.indexOf(raw) !== -1 && t.features[raw]) return t.features[raw];
  if (t.symptoms[base]) return t.symptoms[base].label;
  if (t.comorbidities[base]) return t.comorbidities[base].label;
  if (t.exposure[base]) return t.exposure[base].label;
  if (t.features[base]) return t.features[base];
  return base;
}

function mapNavLang(tag) {
  const t = String(tag || '').toLowerCase();
  if (t.startsWith('zh-tw') || t.startsWith('zh-hk') || t.includes('hant')) return 'zh-TW';
  if (t.startsWith('zh')) return 'zh-CN';
  if (t.startsWith('es')) return 'es';
  if (t.startsWith('pt')) return 'pt';
  return 'en';
}

function detectLang() {
  let saved = null;
  try { saved = localStorage.getItem('lang'); } catch (_) { /* ignore */ }
  if (saved && I18N[saved]) return saved;
  return mapNavLang(navigator.language);
}

function persistLang(code) {
  try { localStorage.setItem('lang', code); } catch (_) { /* ignore */ }
}

function detectMode() {
  let saved = null;
  try { saved = localStorage.getItem('mode'); } catch (_) { /* ignore */ }
  return saved === 'classic' || saved === 'adaptive' ? saved : 'adaptive';
}

function persistMode(mode) {
  try { localStorage.setItem('mode', mode); } catch (_) { /* ignore */ }
}

function markAnswered(kind, code) {
  if (state.answered[kind]) state.answered[kind][code] = true;
}

/* --------------------------------------------------------- View switching */

const VIEWS = ['hero', 'wizard', 'loading', 'result', 'destination'];

function showView(name) {
  VIEWS.forEach((v) => { $(`view-${v}`).hidden = v !== name; });
  if (name !== 'loading') stopLoadingRotation();
  if (name !== 'destination') stopDestRotation();
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

/* --------------------------------------------------------- Language */

function setLanguage(code) {
  if (!I18N[code]) code = DEFAULT_LANG;
  state.lang = code;
  persistLang(code);

  const t = T();
  document.documentElement.lang = code;
  document.title = t.docTitle;
  const meta = document.querySelector('meta[name="description"]');
  if (meta) meta.setAttribute('content', t.metaDescription);

  // Language switcher
  const current = LANGS.find((l) => l.code === code);
  $('lang-current').textContent = current ? current.name : code;
  $('lang-toggle').setAttribute('aria-label', t.langLabel);
  $('lang-menu').setAttribute('aria-label', t.langLabel);
  document.querySelectorAll('.lang-option').forEach((btn) => {
    const on = btn.dataset.lang === code;
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
    btn.classList.toggle('is-current', on);
  });

  // Hero (two entry points: destination lookup / symptom self-assessment)
  $('hero-title').innerHTML = t.hero.title;
  $('btn-destination-title').textContent = t.hero.ctaTravelTitle;
  $('btn-destination-desc').textContent = t.hero.ctaTravelDesc;
  $('btn-start-text').textContent = t.hero.ctaSymptomsTitle;
  $('btn-start-desc').textContent = t.hero.ctaSymptomsDesc;
  t.hero.features.forEach((text, i) => {
    const el = $(`hero-feature-${i + 1}`);
    if (el) el.textContent = text;
  });
  $('btn-privacy').textContent = t.hero.privacyLink;

  // Static parts of the wizard
  $('wizard-brand').textContent = t.brand;
  $('btn-wizard-back-text').textContent = t.nav.back;
  $('btn-prev').textContent = t.nav.prev;
  document.querySelector('.progress').setAttribute('aria-label', t.a11y.progress);
  $('mode-adaptive').textContent = t.mode.adaptive;
  $('mode-classic').textContent = t.mode.classic;
  $('mode-toggle').setAttribute('aria-label', t.mode.switch);

  // Loading

  // Static parts of the result page
  $('result-title').textContent = t.result.title;
  $('label-dengue').textContent = t.result.dengue;
  $('label-severe').textContent = t.result.severe;
  $('label-worsening').textContent = t.result.worsening;
  $('why-dengue').textContent = t.result.explain.why;
  $('why-severe').textContent = t.result.explain.why;
  $('exposure-label').textContent = t.result.exposure.label;
  $('btn-restart').textContent = t.result.restart;
  $('btn-home').textContent = t.result.home;

  // Static parts of the chat panel
  $('chat-title').textContent = t.chat.title;
  $('chat-log').setAttribute('aria-label', t.a11y.chatLog);
  $('chat-input-label').textContent = t.chat.inputLabel;
  $('chat-input').placeholder = t.chat.placeholder;
  $('chat-input').setAttribute('aria-label', t.chat.inputLabel);
  $('chat-send').textContent = t.chat.send;
  $('chat-note').textContent = t.chat.note;
  $('btn-privacy-chat').textContent = t.chat.privacyLink;

  // Error overlay
  $('error-title').textContent = t.errors.title;
  $('btn-retry').textContent = t.errors.retry;
  $('btn-error-close').textContent = t.errors.back;

  // Privacy overlay
  renderPrivacy();

  // Disclaimer (the result page prefers whatever the back end returned)
  $('disclaimer-bar').textContent =
    (lastResult && lastResult.disclaimer) || t.disclaimer;

  // Re-render dynamic content (smart and full modes each restore their own progress)
  if (!$('view-wizard').hidden) renderWizard(null);
  if (!$('view-result').hidden && lastResult) renderResult(lastResult, { animate: false });
  if (!$('error-overlay').hidden && lastError) renderError();
  // Destination view: re-render the whole page in place; both the input and any
  // results already fetched are preserved
  renderDestination();
  renderTravelContext();
  renderChatChips();
  renderChatLog();
}

/* --------------------------------------------------------- Questionnaire rendering */

function answeredCount(step) {
  if (TRI_KINDS.indexOf(step.kind) === -1) return null;
  const bag = state.answers[step.kind];
  const done = step.codes.filter((c) => bag[c] !== 'unknown').length;
  return { done, total: step.codes.length };
}

function makeTriRow(kind, code) {
  const t = T();
  const info = itemText(kind, code);
  const row = document.createElement('div');
  row.className = 'q-row';

  const text = document.createElement('div');
  text.className = 'q-text';
  const label = document.createElement('span');
  label.className = 'q-label';
  label.textContent = info.label;
  const desc = document.createElement('span');
  desc.className = 'q-desc';
  desc.textContent = info.desc;
  text.append(label, desc);

  const opts = document.createElement('div');
  opts.className = 'q-opts';
  opts.setAttribute('role', 'group');
  opts.setAttribute('aria-label', info.label);

  ['yes', 'no', 'unknown'].forEach((value) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `q-opt q-opt-${value}`;
    btn.dataset.value = value;
    btn.textContent = t.answers[value];
    const on = state.answers[kind][code] === value;
    btn.classList.toggle('is-on', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.addEventListener('click', () => {
      state.answers[kind][code] = value;
      markAnswered(kind, code);
      opts.querySelectorAll('.q-opt').forEach((b) => {
        const sel = b.dataset.value === value;
        b.classList.toggle('is-on', sel);
        b.setAttribute('aria-pressed', sel ? 'true' : 'false');
      });
      updateWizardMeta();
    });
    opts.appendChild(btn);
  });

  row.append(text, opts);
  return row;
}

function renderBasicStep(host) {
  const t = T();
  const a = state.answers;

  // Age
  const ageBlock = document.createElement('div');
  ageBlock.className = 'field-block';
  const ageHead = document.createElement('div');
  ageHead.className = 'field-head';
  const ageLabel = document.createElement('span');
  ageLabel.className = 'field-label';
  ageLabel.textContent = t.fields.age;
  const ageValue = document.createElement('span');
  ageValue.className = 'field-value';
  ageValue.textContent = `${a.age} ${t.fields.ageUnit}`;
  ageHead.append(ageLabel, ageValue);
  const ageInput = document.createElement('input');
  ageInput.type = 'range';
  ageInput.className = 'slider';
  ageInput.min = '0';
  ageInput.max = '110';
  ageInput.step = '1';
  ageInput.value = String(a.age);
  ageInput.setAttribute('aria-label', t.fields.age);
  ageInput.addEventListener('input', () => {
    a.age = Number(ageInput.value);
    ageValue.textContent = `${a.age} ${t.fields.ageUnit}`;
  });
  ageBlock.append(ageHead, ageInput);

  // Sex
  const sexBlock = document.createElement('div');
  sexBlock.className = 'field-block';
  const sexLabel = document.createElement('div');
  sexLabel.className = 'field-label';
  sexLabel.textContent = t.fields.sex;
  const sexOpts = document.createElement('div');
  sexOpts.className = 'sex-opts';
  sexOpts.setAttribute('role', 'group');
  sexOpts.setAttribute('aria-label', t.fields.sex);
  [['F', t.fields.sexF, '♀'], ['M', t.fields.sexM, '♂']].forEach(([val, text, glyph]) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sex-opt';
    btn.dataset.value = val;
    const g = document.createElement('span');
    g.className = 'sex-glyph';
    g.textContent = glyph;
    const s = document.createElement('span');
    s.textContent = text;
    btn.append(g, s);
    const on = a.sex === val;
    btn.classList.toggle('is-on', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.addEventListener('click', () => {
      a.sex = val;
      sexOpts.querySelectorAll('.sex-opt').forEach((b) => {
        const sel = b.dataset.value === val;
        b.classList.toggle('is-on', sel);
        b.setAttribute('aria-pressed', sel ? 'true' : 'false');
      });
      hideHint();
    });
    sexOpts.appendChild(btn);
  });
  sexBlock.append(sexLabel, sexOpts);

  // Days of illness
  const dayBlock = document.createElement('div');
  dayBlock.className = 'field-block';
  const dayHead = document.createElement('div');
  dayHead.className = 'field-head';
  const dayLabel = document.createElement('span');
  dayLabel.className = 'field-label';
  dayLabel.textContent = t.fields.dayIll;
  const dayValue = document.createElement('span');
  dayValue.className = 'field-value';
  const dayText = (n) => (n === 0 ? t.fields.dayIllZero : `${n} ${t.fields.dayIllUnit}`);
  dayValue.textContent = dayText(a.dayIll);
  dayHead.append(dayLabel, dayValue);
  const dayInput = document.createElement('input');
  dayInput.type = 'range';
  dayInput.className = 'slider';
  dayInput.min = '0';
  dayInput.max = '14';
  dayInput.step = '1';
  dayInput.value = String(a.dayIll);
  dayInput.setAttribute('aria-label', t.fields.dayIll);
  dayInput.addEventListener('input', () => {
    a.dayIll = Number(dayInput.value);
    dayValue.textContent = dayText(a.dayIll);
  });
  dayBlock.append(dayHead, dayInput);

  host.append(ageBlock, sexBlock, dayBlock);
}

function renderNotesStep(host) {
  const t = T();
  const block = document.createElement('div');
  block.className = 'field-block';

  const ta = document.createElement('textarea');
  ta.className = 'notes-input';
  ta.maxLength = 500;
  ta.rows = 5;
  ta.placeholder = t.notes.placeholder;
  ta.value = state.answers.notes;
  // The step heading is the visible name for this field, so it is also its accessible name
  ta.setAttribute('aria-label', t.steps.notes.title);

  const counter = document.createElement('div');
  counter.className = 'notes-counter';
  const setCount = () => {
    const n = ta.value.length;
    counter.textContent = `${n} / 500`;
    counter.classList.toggle('is-warn', n >= 450);
  };
  ta.addEventListener('input', () => {
    state.answers.notes = ta.value;
    setCount();
  });
  setCount();

  block.append(ta, counter);
  host.appendChild(block);
}

function renderStep(direction) {
  const t = T();
  const step = STEPS[state.step];
  const panel = $('wizard-panel');

  const wrap = document.createElement('div');
  wrap.className = 'step';
  if (direction === 'forward') wrap.classList.add('slide-in-right');
  else if (direction === 'back') wrap.classList.add('slide-in-left');

  // Heading
  const head = stepHead(t.steps[step.id].title, t.steps[step.id].sub);
  const h2 = head.querySelector('.step-title');
  wrap.appendChild(head);

  if (step.kind === 'basic') {
    renderBasicStep(wrap);
  } else if (step.kind === 'notes') {
    renderNotesStep(wrap);
  } else {
    const list = document.createElement('div');
    list.className = 'q-list';
    step.codes.forEach((code) => list.appendChild(makeTriRow(step.kind, code)));
    wrap.appendChild(list);
  }

  panel.replaceChildren(wrap);
  updateStepMeta();
  hideHint();

  // Navigation buttons (smart mode may have hidden them; restore the full
  // questionnaire's layout here)
  document.querySelector('.wizard-nav').hidden = false;
  $('btn-prev').hidden = false;
  $('btn-prev').disabled = state.step === 0;
  const isLast = state.step === STEPS.length - 1;
  $('btn-next').textContent = isLast ? t.nav.submit : t.nav.next;

  h2.focus({ preventScroll: true });
}

function updateStepMeta() {
  const t = T();
  const step = STEPS[state.step];
  const total = STEPS.length;
  const cur = state.step + 1;

  let counter = fmt(t.stepCounter, { cur, total });
  const counts = answeredCount(step);
  if (counts) {
    counter += ` · ${fmt(t.answered, { n: counts.done, total: counts.total })}`;
  }
  $('step-counter').textContent = counter;

  $('progress-fill').style.width = `${(cur / total) * 100}%`;
  const bar = document.querySelector('.progress');
  bar.setAttribute('aria-valuenow', String(cur));
  bar.setAttribute('aria-valuemax', String(total));

  const dots = $('progress-steps');
  dots.replaceChildren(...STEPS.map((s, i) => {
    const dot = document.createElement('span');
    dot.className = 'progress-dot';
    if (i < cur) dot.classList.add('is-done');
    if (i === state.step) dot.classList.add('is-current');
    return dot;
  }));
}

function showHint(text) {
  const el = $('wizard-hint');
  el.textContent = text;
  el.hidden = false;
  el.classList.remove('shake');
  void el.offsetWidth;
  el.classList.add('shake');
}

function hideHint() {
  const el = $('wizard-hint');
  el.hidden = true;
  el.textContent = '';
}

function validateStep() {
  const step = STEPS[state.step];
  if (step.kind === 'basic' && !state.answers.sex) {
    showHint(T().hints.sexRequired);
    return false;
  }
  return true;
}

function goStep(index, direction) {
  state.step = Math.max(0, Math.min(STEPS.length - 1, index));
  renderStep(direction);
}

/* --------------------------------------------------------- Smart interview (adaptive mode) */
/*
 * The /api/plan contract: the request carries only the questions that have been
 * answered (yes/no/unknown all count as answered, and keys never asked must be
 * absent); the response gives bounds -- the upper and lower score bounds for the three
 * models -- plus can_stop and up to 5 candidate next questions in next.
 * Any failure (including a 6-second timeout, or a 404) silently falls back to the full
 * questionnaire, carrying every answer across -- smart mode is only an enhancement and
 * must never get in the user's way.
 */

function renderWizard(direction) {
  if (state.mode === 'adaptive') renderAdaptive(direction);
  else renderStep(direction);
}

function updateWizardMeta() {
  if (state.mode === 'adaptive') updateAdaptiveMeta();
  else updateStepMeta();
}

function updateModeToggle() {
  const on = state.mode === 'adaptive';
  const a = $('mode-adaptive');
  const c = $('mode-classic');
  a.classList.toggle('is-on', on);
  a.setAttribute('aria-pressed', on ? 'true' : 'false');
  c.classList.toggle('is-on', !on);
  c.setAttribute('aria-pressed', on ? 'false' : 'true');
}

/**
 * Switch interview mode. Answers are fully shared: the full questionnaire shows what
 * has been answered, and switching back to smart mode re-plans from there.
 * opts.persist=false is for a silent fallback (not the user's own choice, so it is not
 * written to their preferences).
 */
function setMode(mode, opts) {
  if (mode !== 'adaptive' && mode !== 'classic') mode = 'adaptive';
  state.mode = mode;
  if (!opts || opts.persist !== false) persistMode(mode);
  updateModeToggle();
  if ($('view-wizard').hidden) return;
  if (mode === 'adaptive') enterAdaptive();
  else renderStep(null);
}

/** Decide which stage of smart mode to enter, based on what has been answered. */
function computeAdaptiveStage() {
  if (!state.answers.sex) return 'basic';
  if (!SAFETY_CODES.every((c) => state.answered.symptoms[c])) return 'safety';
  if (!EXPOSURE_CODES.every((c) => state.answered.exposure[c])) return 'exposure';
  return 'loop';
}

function enterAdaptive() {
  const ad = state.adaptive;
  ad.stage = computeAdaptiveStage();
  ad.current = null;
  ad.stopReason = null;
  renderAdaptive(null);
  if (ad.stage === 'loop') advancePlan();
}

/** The /api/plan request body: only the questions the user has genuinely answered. */
function buildPlanPayload() {
  const a = state.answers;
  const symptoms = {};
  SYMPTOM_CODES.forEach((c) => {
    if (state.answered.symptoms[c]) symptoms[c] = a.symptoms[c];
  });
  const comorbidities = {};
  COMORB_CODES.forEach((c) => {
    if (state.answered.comorbidities[c]) comorbidities[c] = a.comorbidities[c];
  });
  return {
    age: a.age,
    sex: a.sex,
    day_ill: a.dayIll,
    symptoms,
    comorbidities,
    language: state.lang,
  };
}

async function fetchPlan() {
  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  const timer = ctrl ? setTimeout(() => ctrl.abort(), PLAN_TIMEOUT_MS) : null;
  try {
    const resp = await fetch('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPlanPayload()),
      signal: ctrl ? ctrl.signal : undefined,
    });
    if (!resp.ok) throw new Error(`plan status ${resp.status}`);
    const data = await resp.json();
    if (!data || typeof data !== 'object' || !data.bounds) throw new Error('plan bad data');
    return data;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Pick the first question from next that we recognise and that is not yet answered. */
function pickNext(plan) {
  const list = Array.isArray(plan.next) ? plan.next : [];
  for (let i = 0; i < list.length; i += 1) {
    const it = list[i];
    if (!it || typeof it !== 'object') continue;
    const kind = PLAN_KIND_MAP[it.kind];
    if (!kind) continue;
    const code = String(it.code || '');
    const known = kind === 'symptoms' ? SYMPTOM_CODES : COMORB_CODES;
    if (known.indexOf(code) === -1) continue;
    if (state.answered[kind][code]) continue;
    return {
      kind,
      code,
      why: METRICS.indexOf(it.why_model) !== -1 ? it.why_model : null,
    };
  }
  return null;
}

/** Step heading. subText is optional: a step with no subtitle gets no empty paragraph. */
function stepHead(title, subText) {
  const head = document.createElement('header');
  head.className = 'step-head';
  const h2 = document.createElement('h2');
  h2.className = 'step-title';
  h2.tabIndex = -1;
  h2.textContent = title;
  head.appendChild(h2);
  if (subText) {
    const sub = document.createElement('p');
    sub.className = 'step-sub';
    sub.textContent = subText;
    head.appendChild(sub);
  }
  return head;
}

/** Render the current question card; with no question, show loading dots or nothing,
 *  depending on the planning state. */
function renderQuestion(focusQuestion) {
  const host = $('aq-host');
  if (!host) return;
  const t = T();
  const ad = state.adaptive;

  if (!ad.current) {
    if (ad.planning) {
      const load = document.createElement('div');
      load.className = 'aq-loading';
      for (let i = 0; i < 3; i += 1) {
        const dot = document.createElement('span');
        dot.className = 'dot';
        dot.setAttribute('aria-hidden', 'true');
        load.appendChild(dot);
      }
      host.replaceChildren(load);
    } else {
      host.replaceChildren();
    }
    return;
  }

  const q = ad.current;
  const info = itemText(q.kind, q.code);

  const card = document.createElement('div');
  card.className = 'aq-card';

  const meta = document.createElement('p');
  meta.className = 'aq-meta';
  let metaText = fmt(t.adaptive.questionCounter, { n: ad.loopCount + 1 });
  meta.textContent = metaText;
  card.appendChild(meta);

  const label = document.createElement('h3');
  label.className = 'aq-label';
  label.tabIndex = -1;
  label.textContent = info.label;
  card.appendChild(label);

  const opts = document.createElement('div');
  opts.className = 'aq-opts';
  opts.setAttribute('role', 'group');
  opts.setAttribute('aria-label', info.label);
  ['yes', 'no', 'unknown'].forEach((value) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `aq-opt aq-opt-${value}`;
    btn.textContent = t.answers[value];
    btn.addEventListener('click', () => answerCurrent(value));
    opts.appendChild(btn);
  });
  card.appendChild(opts);

  
  host.replaceChildren(card);
  if (focusQuestion) label.focus({ preventScroll: true });
}

function setQuestionBusy(busy) {
  document.querySelectorAll('.aq-opt').forEach((b) => { b.disabled = busy; });
}

/** Answer the current question (by click or by pressing 1/2/3), then re-plan. */
function answerCurrent(value) {
  const ad = state.adaptive;
  if (ad.stage !== 'loop' || !ad.current || ad.planning) return;
  const q = ad.current;
  state.answers[q.kind][q.code] = value;
  markAnswered(q.kind, q.code);
  ad.loopCount += 1;
  // Note: current is kept until the new plan arrives; disabling the old card is enough,
  // and it avoids a flash of loading dots
  advancePlan();
}

/** Request /api/plan and advance: update the tracks, then either stop or show the next
 *  question. */
async function advancePlan() {
  const ad = state.adaptive;
  ad.planSeq += 1;
  const seq = ad.planSeq;
  ad.planning = true;
  setQuestionBusy(true);
  if (!ad.current) renderQuestion(false); // just entered the loop: show loading dots

  let plan = null;
  try {
    plan = await fetchPlan();
  } catch (_) {
    // /api/plan unavailable (404, timeout, malformed response...): silently fall back
    // to the full questionnaire
    if (state.adaptive === ad && seq === ad.planSeq) {
      ad.planning = false;
      if (state.mode === 'adaptive' && ad.stage === 'loop') fallbackToClassic();
    }
    return;
  }

  // Discard stale responses (the user may have reset, switched mode, or changed
  // language and re-requested in the meantime)
  if (state.adaptive !== ad || seq !== ad.planSeq) return;
  ad.planning = false;
  if (state.mode !== 'adaptive' || ad.stage !== 'loop') return;

  const q = pickNext(plan);
  if (plan.can_stop === true) { stopLoop('proven'); return; }
  if (ad.loopCount >= ADAPTIVE_CAP) { stopLoop('cap'); return; }
  if (!q) { stopLoop('nomore'); return; }

  ad.current = q;
  renderQuestion(true);
}

function stopLoop(reason) {
  const ad = state.adaptive;
  ad.stopReason = reason;
  ad.current = null;
  ad.stage = 'notes';
  renderAdaptive('forward');
}

/** Silent degradation when smart mode fails: keep every answer and move to the full
 *  questionnaire. */
function fallbackToClassic() {
  state.adaptive.current = null;
  state.adaptive.planning = false;
  if (state.answers.sex) state.step = Math.max(state.step, 1);
  setMode('classic', { persist: false });
}

function adaptiveNext() {
  const ad = state.adaptive;
  if (ad.stage === 'basic') {
    if (!state.answers.sex) { showHint(T().hints.sexRequired); return; }
    ad.stage = 'safety';
    renderAdaptive('forward');
  } else if (ad.stage === 'safety') {
    ad.stage = 'exposure';
    renderAdaptive('forward');
  } else if (ad.stage === 'exposure') {
    ad.stage = 'loop';
    ad.current = null;
    ad.stopReason = null;
    renderAdaptive('forward');
    advancePlan();
  } else if (ad.stage === 'notes') {
    submit();
  }
}

function adaptivePrev() {
  const ad = state.adaptive;
  if (ad.stage === 'safety') {
    ad.stage = 'basic';
    renderAdaptive('back');
  } else if (ad.stage === 'exposure') {
    ad.stage = 'safety';
    renderAdaptive('back');
  }
}

function updateAdaptiveMeta() {
  const t = T();
  const ad = state.adaptive;
  const idx = Math.max(0, ADAPTIVE_STAGES.indexOf(ad.stage));
  const total = ADAPTIVE_STAGES.length;
  const cur = idx + 1;

  let counter = fmt(t.stepCounter, { cur, total });
  let codes = null;
  let kind = null;
  if (ad.stage === 'safety') { codes = SAFETY_CODES; kind = 'symptoms'; }
  else if (ad.stage === 'exposure') { codes = EXPOSURE_CODES; kind = 'exposure'; }
  if (codes) {
    const done = codes.filter((c) => state.answered[kind][c]).length;
    counter += ` · ${fmt(t.answered, { n: done, total: codes.length })}`;
  }
  $('step-counter').textContent = counter;

  $('progress-fill').style.width = `${(cur / total) * 100}%`;
  const bar = document.querySelector('.progress');
  bar.setAttribute('aria-valuenow', String(cur));
  bar.setAttribute('aria-valuemax', String(total));

  const dots = $('progress-steps');
  dots.replaceChildren(...ADAPTIVE_STAGES.map((s, i) => {
    const dot = document.createElement('span');
    dot.className = 'progress-dot';
    if (i < cur) dot.classList.add('is-done');
    if (i === idx) dot.classList.add('is-current');
    return dot;
  }));
}

/** Render the current stage of smart mode (a language change comes through here too,
 *  with all state preserved). */
function renderAdaptive(direction) {
  const t = T();
  const ad = state.adaptive;
  const panel = $('wizard-panel');

  const wrap = document.createElement('div');
  wrap.className = 'step';
  if (direction === 'forward') wrap.classList.add('slide-in-right');
  else if (direction === 'back') wrap.classList.add('slide-in-left');

  let head = null;

  if (ad.stage === 'basic') {
    head = stepHead(t.steps.basic.title, t.steps.basic.sub);
    wrap.appendChild(head);
    renderBasicStep(wrap);
  } else if (ad.stage === 'safety') {
    // VOMITO / PETEQUIA_N are WHO warning signs: always asked, of everyone
    head = stepHead(t.adaptive.stageSafety, null);
    wrap.appendChild(head);
    const list = document.createElement('div');
    list.className = 'q-list';
    SAFETY_CODES.forEach((code) => list.appendChild(makeTriRow('symptoms', code)));
    wrap.appendChild(list);
  } else if (ad.stage === 'exposure') {
    // Exactly the same source as step 5 of the full questionnaire: same codes, same copy
    head = stepHead(t.steps.exposure.title, t.steps.exposure.sub);
    wrap.appendChild(head);
    const list = document.createElement('div');
    list.className = 'q-list';
    EXPOSURE_CODES.forEach((code) => list.appendChild(makeTriRow('exposure', code)));
    wrap.appendChild(list);
  } else if (ad.stage === 'loop') {
    head = stepHead(t.adaptive.stageLoop, null);
    wrap.appendChild(head);
    const host = document.createElement('div');
    host.id = 'aq-host';
    wrap.appendChild(host);
  } else {
    // notes: stop reason + free-text notes + the way into the full questionnaire
    head = stepHead(t.steps.notes.title, t.steps.notes.sub);
    wrap.appendChild(head);
    renderNotesStep(wrap);
    const more = document.createElement('p');
    more.className = 'aq-more';
    const moreBtn = document.createElement('button');
    moreBtn.type = 'button';
    moreBtn.className = 'link-btn';
    moreBtn.textContent = t.adaptive.continueFull;
    moreBtn.addEventListener('click', () => {
      // Answering more means switching to the full questionnaire; this is not a mode
      // preference and is not written to localStorage
      if (state.answers.sex) state.step = Math.max(state.step, 1);
      setMode('classic', { persist: false });
    });
    more.appendChild(moreBtn);
    wrap.appendChild(more);
  }

  panel.replaceChildren(wrap);

  // The question card must be filled in after the DOM is mounted (it relies on
  // getElementById)
  if (ad.stage === 'loop') renderQuestion(false);

  updateAdaptiveMeta();
  hideHint();

  // Navigation: during the loop stage the answer buttons drive it, so hide the whole
  // navigation bar
  const nav = document.querySelector('.wizard-nav');
  nav.hidden = ad.stage === 'loop';
  const prev = $('btn-prev');
  prev.hidden = ad.stage === 'notes';
  prev.disabled = ad.stage === 'basic';
  $('btn-next').textContent = ad.stage === 'notes' ? t.nav.submit : t.nav.next;

  if (head) head.querySelector('.step-title').focus({ preventScroll: true });
}

/* --------------------------------------------------------- Submission */

function buildPayload() {
  const a = state.answers;
  return {
    age: a.age,
    sex: a.sex,
    day_ill: a.dayIll,
    symptoms: { ...a.symptoms },
    comorbidities: { ...a.comorbidities },
    // Takes no part in model scoring; the back end turns it into exposure_context by rule
    exposure: { ...a.exposure },
    language: state.lang,
    notes: a.notes.trim().slice(0, 500),
  };
}

function startLoadingRotation() {
  const msgs = T().loading.steps;
  let i = 0;
  $('loading-text').textContent = msgs[0];
  stopLoadingRotation();
  loadingTimer = setInterval(() => {
    i = (i + 1) % msgs.length;
    $('loading-text').textContent = T().loading.steps[i];
  }, 1500);
}

function stopLoadingRotation() {
  if (loadingTimer) {
    clearInterval(loadingTimer);
    loadingTimer = null;
  }
}

async function submit() {
  if (state.submitting) return;
  state.submitting = true;
  showView('loading');
  startLoadingRotation();

  const minWait = new Promise((r) => setTimeout(r, 2000));
  try {
    const resp = await fetch('/api/assess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    });

    if (!resp.ok) {
      let detail = null;
      try {
        const body = await resp.json();
        if (typeof body.detail === 'string') detail = body.detail;
      } catch (_) { /* ignore parse failures */ }
      await minWait;
      lastError = { status: resp.status, detail };
      showErrorOverlay();
      return;
    }

    const data = await resp.json();
    await minWait;
    if (!data || !data.dengue || !data.advice) {
      lastError = { kind: 'badData' };
      showErrorOverlay();
      return;
    }
    lastResult = data;
    // A new result means a new session: clear the chat log and the expanded state
    state.chat = freshChat();
    METRICS.forEach((m) => { state.openExplain[m] = false; });
    $('chat-input').value = '';
    updateChatCounter();
    showView('result');
    renderResult(data, { animate: true });
  } catch (_) {
    await minWait;
    lastError = { kind: 'network' };
    showErrorOverlay();
  } finally {
    state.submitting = false;
    stopLoadingRotation();
  }
}

/* --------------------------------------------------------- Result rendering */

const GAUGE_R = 92;
const GAUGE_C = 2 * Math.PI * GAUGE_R;

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function animateGauge(arcEl, scoreEl, target, animate) {
  const clamped = Math.max(0, Math.min(100, Number(target) || 0));
  const finalOffset = GAUGE_C * (1 - clamped / 100);
  arcEl.style.strokeDasharray = String(GAUGE_C);

  // Set the final value first: even if animation frames never run (background tab,
  // throttled environment), what is displayed is still the correct result
  arcEl.style.strokeDashoffset = String(finalOffset);
  scoreEl.textContent = clamped.toFixed(1);

  if (!animate || prefersReducedMotion()) {
    arcEl.style.transition = 'none';
    return;
  }

  // The ring is driven by a CSS transition (no rAF dependency): zero it, force a
  // reflow, then set the final value
  arcEl.style.transition = 'none';
  arcEl.style.strokeDashoffset = String(GAUGE_C);
  void arcEl.getBoundingClientRect();
  arcEl.style.transition = 'stroke-dashoffset 1.1s cubic-bezier(0.22, 1, 0.36, 1)';
  arcEl.style.strokeDashoffset = String(finalOffset);

  // The number counts up via rAF; if that never runs, the final value set above stands
  const duration = 1100;
  const start = performance.now();
  const ease = (p) => 1 - Math.pow(1 - p, 3);
  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    scoreEl.textContent = (clamped * ease(p)).toFixed(1);
    if (p < 1) requestAnimationFrame(tick);
    else scoreEl.textContent = clamped.toFixed(1);
  };
  requestAnimationFrame(tick);
}

function renderResult(data, { animate }) {
  const t = T();

  // The two ring gauges
  const gauges = [
    { wrap: 'gauge-dengue', arc: 'arc-dengue', score: 'score-dengue', badge: 'badge-dengue', d: data.dengue },
    { wrap: 'gauge-severe', arc: 'arc-severe', score: 'score-severe', badge: 'badge-severe', d: data.severe },
  ];
  gauges.forEach(({ wrap, arc, score, badge, d }) => {
    const w = $(wrap);
    w.classList.remove('level-low', 'level-medium', 'level-high');
    w.classList.add(`level-${d.level}`);
    $(badge).textContent = t.result.levels[d.level];
    animateGauge($(arc), $(score), d.score, animate);
  });

  // The worsening-risk bar
  const worse = data.worsening;
  const metric = $('metric-worsening');
  metric.classList.remove('level-low', 'level-medium', 'level-high');
  metric.classList.add(`level-${worse.level}`);
  $('score-worsening').textContent = Number(worse.score).toFixed(1);
  $('badge-worsening').textContent = t.result.levels[worse.level];
  const fill = $('bar-worsening');
  if (animate && !prefersReducedMotion()) {
    // Same "zero, force reflow, set final value" CSS transition; no rAF dependency
    fill.style.transition = 'none';
    fill.style.width = '0%';
    void fill.getBoundingClientRect();
    fill.style.transition = '';
  }
  fill.style.width = `${worse.score}%`;

  // WHO warning-sign banner (rule-based, independent of the model score)
  const banner = $('warning-banner');
  const signs = Array.isArray(data.warning_signs) ? data.warning_signs : [];
  if (signs.length) {
    const names = signs.map((c) => itemText('symptoms', c).label).join(t.result.warning.sep);
    $('warning-title').textContent = t.result.warning.title;
    $('warning-text').textContent = fmt(t.result.warning.body, { signs: names });
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }

  // Text
  $('result-summary').textContent = data.summary || '';
  $('model-note').textContent = data.model_note || '';
  $('disclaimer-bar').textContent = data.disclaimer || t.disclaimer;

  // Epidemiological exposure context (rule-based)
  renderExposure(data.exposure_context);

  // Model contribution panels (expandable)
  renderAllExplanations(data.explanations);

  // Travel context (from the destination lookup; took no part in the scoring)
  renderTravelContext();

  // Advice cards: seek care -> monitoring -> protection
  fillAdviceCards($('advice-cards'), data.advice, animate);

  // Follow-up chat
  renderChatChips();
  renderChatLog();
}

/** Advice cards (seek care -> monitoring -> protection); the result page and the
 *  destination page share one card style. */
function fillAdviceCards(host, advice, animate) {
  const t = T();
  host.replaceChildren();
  ADVICE_ORDER.forEach((key, idx) => {
    const items = (advice && advice[key]) || [];
    if (!Array.isArray(items) || !items.length) return;
    const card = document.createElement('section');
    card.className = 'card advice-card';
    if (animate && !prefersReducedMotion()) {
      card.classList.add('fade-up');
      card.style.animationDelay = `${0.12 * idx}s`;
    } else {
      card.style.opacity = '1';
    }
    const h3 = document.createElement('h3');
    h3.className = 'advice-title';
    h3.textContent = t.result.advice[key];
    const ul = document.createElement('ul');
    ul.className = 'advice-list';
    items.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = String(item);
      ul.appendChild(li);
    });
    card.append(h3, ul);
    host.appendChild(card);
  });
  return host.childElementCount > 0;
}

/* --------------------------------------------------------- Exposure context */

function renderExposure(ctx) {
  const t = T().result.exposure;
  const chip = $('exposure-chip');
  const level = ctx && ctx.level;

  if (!level || !t.levels[level]) {
    chip.hidden = true;
    return;
  }

  chip.classList.remove('exp-low', 'exp-medium', 'exp-high');
  chip.classList.add(`exp-${level}`);
  $('exposure-label').textContent = t.label;
  $('exposure-level').textContent = t.levels[level];

  const factors = Array.isArray(ctx.factors) ? ctx.factors : [];
  const names = factors.map((c) => itemText('exposure', c).label);
  $('exposure-factors').textContent = names.length
    ? fmt(t.factors, { list: names.join(t.sep) })
    : t.none;
  chip.hidden = false;
}

/* ------------------------------------- Source links (shared by destination and chat) */

/**
 * Accept http(s) links only. The back end currently only ever produces ordinary URLs,
 * but anything rendered as a clickable link has to be vetted here: a scheme such as
 * javascript: must never reach an href.
 */
function safeUrl(value) {
  const raw = String(value == null ? '' : value).trim();
  if (!raw) return null;
  try {
    const parsed = new URL(raw, window.location.href);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
    return parsed.href;
  } catch (_) {
    return null;
  }
}

/** One source: rendered as a real link opening in a new tab when it has a usable URL,
 *  and degrading to plain text otherwise. */
function sourceItem(item) {
  const url = safeUrl(item.url);
  const title = String(item.title == null ? '' : item.title).trim() || url;
  if (!title) return null;

  const li = document.createElement('li');
  li.className = 'src-item';

  if (url) {
    const a = document.createElement('a');
    a.className = 'src-link';
    a.href = url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = title;
    li.appendChild(a);
  } else {
    const span = document.createElement('span');
    span.className = 'src-plain';
    span.textContent = title;
    li.appendChild(span);
  }

  // Search results return health authorities and news sites side by side. The back end
  // has already judged authority by domain; marking the government / international health
  // bodies here is what lets the reader tell which ones are more checkable.
  // The WHO group already says "World Health Organization" on the label, so hanging an
  // "official" badge on it as well would be redundant.
  if (item.authority === 'official' && item.origin !== 'who') {
    const badge = document.createElement('span');
    badge.className = 'src-official';
    badge.textContent = T().sources.official;
    li.appendChild(badge);
  }

  const date = String(item.date == null ? '' : item.date).trim();
  if (date) {
    const d = document.createElement('span');
    d.className = 'src-date';
    d.textContent = date;
    li.appendChild(d);
  }
  return li;
}

/**
 * The sources block: split by origin into a "WHO official" group and a "web search" group.
 * Returns null when sources is empty -- callers must not render an empty heading.
 */
function buildSourcesBlock(sources, className) {
  const t = T();
  const list = Array.isArray(sources) ? sources.filter((s) => s && typeof s === 'object') : [];
  if (!list.length) return null;

  const groups = [
    { key: 'who', label: t.sources.who, items: list.filter((s) => s.origin === 'who') },
    { key: 'search', label: t.sources.search, items: list.filter((s) => s.origin !== 'who') },
  ];

  const wrap = document.createElement('div');
  wrap.className = className;

  const title = document.createElement('p');
  title.className = 'src-title';
  title.textContent = t.sources.label;
  wrap.appendChild(title);

  let rendered = 0;
  groups.forEach((g) => {
    if (!g.items.length) return;
    const nodes = g.items.map(sourceItem).filter(Boolean);
    if (!nodes.length) return;

    const group = document.createElement('div');
    group.className = 'src-group';
    const tag = document.createElement('span');
    tag.className = `src-tag tag-${g.key}`;
    tag.textContent = g.label;
    const ul = document.createElement('ul');
    ul.className = 'src-list';
    nodes.forEach((n) => ul.appendChild(n));
    group.append(tag, ul);
    wrap.appendChild(group);
    rendered += nodes.length;
  });

  return rendered ? wrap : null;
}

/* --------------------------------------------------------- Destination lookup */

function destErrorMessage() {
  const t = T().dest.errors;
  const err = state.destination.error;
  if (!err) return t.server;
  if (err.kind === 'network') return t.network;
  if (err.kind === 'badData') return t.badData;
  if (err.detail) return err.detail;
  const s = err.status;
  if (s >= 500) return t.server;
  return fmt(t.generic, { status: s });
}

function showDestHint(text) {
  state.destination.hint = true;
  const el = $('dest-hint');
  el.textContent = text;
  el.hidden = false;
}

function hideDestHint() {
  state.destination.hint = false;
  const el = $('dest-hint');
  el.hidden = true;
  el.textContent = '';
}

function startDestRotation() {
  const msgs = T().dest.loading.steps;
  let i = 0;
  $('dest-loading-text').textContent = msgs[0];
  stopDestRotation();
  destLoadingTimer = setInterval(() => {
    i = (i + 1) % msgs.length;
    $('dest-loading-text').textContent = T().dest.loading.steps[i];
  }, 1800);
}

function stopDestRotation() {
  if (destLoadingTimer) {
    clearInterval(destLoadingTimer);
    destLoadingTimer = null;
  }
}

function renderDestChips() {
  const t = T().dest;
  const host = $('dest-chips');
  host.replaceChildren(...t.examples.map((name) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'dest-chip';
    btn.textContent = name;
    btn.disabled = state.destination.loading;
    btn.addEventListener('click', () => {
      state.destination.query = name;
      $('dest-input').value = name;
      runDestination(name);
    });
    return btn;
  }));
}

/** One information-layer card: icon + heading + trustworthiness label + body.
 *  The card's own styling (solid vs dashed) is what carries the hierarchy. */
function buildLayerCard(variant, icon, title, tag, bodyNodes) {
  const sec = document.createElement('section');
  sec.className = `dest-layer layer-${variant}`;

  const head = document.createElement('header');
  head.className = 'dest-layer-head';
  const ic = document.createElement('span');
  ic.className = 'dest-layer-icon';
  ic.setAttribute('aria-hidden', 'true');
  ic.textContent = icon;
  const headings = document.createElement('div');
  headings.className = 'dest-layer-headings';
  const h = document.createElement('h4');
  h.className = 'dest-layer-title';
  h.textContent = title;
  const tg = document.createElement('span');
  tg.className = 'dest-layer-tag';
  tg.textContent = tag;
  headings.append(h, tg);
  head.append(ic, headings);
  sec.appendChild(head);

  bodyNodes.forEach((n) => { if (n) sec.appendChild(n); });
  return sec;
}

function destEmptyLine(text) {
  const p = document.createElement('p');
  p.className = 'dest-empty';
  p.textContent = text;
  return p;
}

/** Layer one: regional background (endemicity + seasonality). Qualitative badges only;
 *  a number must never appear here. */
function buildReferenceLayer(data) {
  const t = T().dest;
  const matched = data.matched === true;
  const season = String(data.season_note == null ? '' : data.season_note).trim();
  if (!matched && !season) return null;

  const body = [];

  if (matched) {
    const level = ENDEMICITY_LEVELS.indexOf(data.endemicity) !== -1 ? data.endemicity : 'unknown';
    const row = document.createElement('p');
    row.className = 'dest-endemicity';
    const label = document.createElement('span');
    label.className = 'dest-endemicity-label';
    label.textContent = t.endemicity.label;
    const badge = document.createElement('span');
    badge.className = `endem-badge endem-${level}`;
    badge.textContent = t.endemicity.levels[level];
    row.append(label, badge);
    body.push(row);
  }

  if (season) {
    const p = document.createElement('p');
    p.className = 'dest-season';
    const label = document.createElement('span');
    label.className = 'dest-season-label';
    label.textContent = t.seasonLabel;
    p.append(label, document.createTextNode(season));
    body.push(p);
  }

  return buildLayerCard(
    'reference', '📚', t.layers.reference.title, t.layers.reference.tag, body,
  );
}

/** Layer two: WHO notices. Every one carries its real date -- some are years old, and
 *  the date is the honesty. */
function buildWhoLayer(data) {
  const t = T().dest;
  const notices = Array.isArray(data.who_notices)
    ? data.who_notices.filter((n) => n && typeof n === 'object')
    : [];

  let body;
  if (!notices.length) {
    body = [destEmptyLine(t.layers.who.empty)];
  } else {
    const ul = document.createElement('ul');
    ul.className = 'dest-notices';
    notices.forEach((n) => {
      const url = safeUrl(n.url);
      const title = String(n.title == null ? '' : n.title).trim() || url;
      if (!title) return;

      const li = document.createElement('li');
      li.className = 'notice-item';

      const date = String(n.date == null ? '' : n.date).trim();
      if (date) {
        const d = document.createElement('span');
        d.className = 'notice-date';
        d.textContent = date;
        li.appendChild(d);
      }

      // The link itself is the flex child of this row, so the touch target fills the
      // whole line
      let titleEl;
      if (url) {
        titleEl = document.createElement('a');
        titleEl.className = 'notice-title src-link';
        titleEl.href = url;
        titleEl.target = '_blank';
        titleEl.rel = 'noopener noreferrer';
      } else {
        titleEl = document.createElement('span');
        titleEl.className = 'notice-title src-plain';
      }
      titleEl.textContent = title;
      li.appendChild(titleEl);
      ul.appendChild(li);
    });
    body = ul.childElementCount ? [ul] : [destEmptyLine(t.layers.who.empty)];
  }

  return buildLayerCard(
    'who', '🏛️', t.layers.who.title, t.layers.who.tag, body,
  );
}

/** Layer three: recent reporting from the model's web search. May be empty; when search
 *  is degraded or disabled, say so plainly. */
function buildRecentLayer(data) {
  const t = T().dest;
  const findings = Array.isArray(data.recent_findings)
    ? data.recent_findings
      .map((f) => String(f == null ? '' : f).trim())
      .filter((f) => f)
    : [];

  const body = [];
  if (findings.length) {
    const ul = document.createElement('ul');
    ul.className = 'dest-findings';
    findings.forEach((f) => {
      const li = document.createElement('li');
      li.textContent = f;
      ul.appendChild(li);
    });
    body.push(ul);
  } else {
    body.push(destEmptyLine(t.layers.recent.empty));
  }

  const status = data.search_status;
  if (status === 'degraded' || status === 'disabled') {
    const note = document.createElement('p');
    note.className = 'dest-status';
    note.textContent = t.status[status];
    body.push(note);
  }

  return buildLayerCard(
    'recent', '🔎', t.layers.recent.title, t.layers.recent.tag, body,
  );
}

function renderDestResults() {
  const t = T();
  const d = state.destination;
  const host = $('dest-results');
  host.replaceChildren();

  if (!d.data || d.loading) {
    host.hidden = true;
    return;
  }
  const data = d.data;
  host.hidden = false;

  const location = String(data.location == null ? '' : data.location).trim() || d.location;

  const title = document.createElement('h3');
  title.className = 'dest-result-title';
  title.id = 'dest-result-title';
  title.tabIndex = -1;
  title.textContent = fmt(t.dest.resultTitle, { location });
  host.appendChild(title);

  // No match in the reference table: say so, and do not guess the endemicity
  if (data.matched !== true) {
    const un = document.createElement('p');
    un.className = 'dest-unmatched';
    un.textContent = fmt(t.dest.notMatched, { location });
    host.appendChild(un);
  }

  // The three layers, in descending order of trustworthiness, presented as blocks
  const reference = buildReferenceLayer(data);
  if (reference) host.appendChild(reference);
  host.appendChild(buildWhoLayer(data));
  host.appendChild(buildRecentLayer(data));

  // Advice: seek care -> monitoring -> protection
  const grid = document.createElement('div');
  grid.className = 'advice-grid';
  if (fillAdviceCards(grid, data.advice, false)) host.appendChild(grid);

  // Sources: WHO official / web search, labelled by group, all opening in a new tab
  const sources = buildSourcesBlock(data.sources, 'dest-sources');
  if (sources) host.appendChild(sources);
}

/** Render entry point for the whole destination view; on a language change it re-runs in
 *  place, losing neither the input nor the results. */
function renderDestination() {
  const t = T();
  const d = state.destination;

  $('dest-title').textContent = t.dest.title;
  $('dest-sub').textContent = t.dest.sub;
  $('btn-dest-back-text').textContent = t.dest.back;
  $('dest-input-label').textContent = t.dest.inputLabel;
  $('dest-submit').textContent = t.dest.submit;
  $('dest-examples-label').textContent = t.dest.examplesLabel;
  $('btn-dest-to-symptoms').textContent = t.dest.toSymptoms;
  $('dest-retry').textContent = t.dest.errors.retry;
  $('dest-disclaimer-bar').textContent = (d.data && d.data.disclaimer) || t.disclaimer;

  const input = $('dest-input');
  input.placeholder = t.dest.placeholder;
  input.setAttribute('aria-label', t.dest.inputLabel);
  input.disabled = d.loading;
  if (input.value !== d.query) input.value = d.query;
  $('dest-submit').disabled = d.loading;

  renderDestChips();

  // The empty-input hint: a single sentence, rewritten to follow the current language
  if (d.hint) showDestHint(t.dest.emptyHint);
  else hideDestHint();

  // Errors are shown in place with a retry; they do not go through the global error overlay
  const errBox = $('dest-error');
  if (d.error && !d.loading) {
    $('dest-error-text').textContent = destErrorMessage();
    errBox.hidden = false;
  } else {
    errBox.hidden = true;
  }

  $('dest-loading').hidden = !d.loading;
  if (d.loading) startDestRotation();
  else stopDestRotation();

  renderDestResults();
}

function openDestination(from) {
  state.destination.from = from === 'result' ? 'result' : 'hero';
  renderDestination();
  showView('destination');
  $('dest-input').focus({ preventScroll: true });
}

function leaveDestination() {
  const d = state.destination;
  if (d.from === 'result' && lastResult) {
    renderTravelContext();
    showView('result');
    return;
  }
  showView('hero');
}

async function runDestination(location) {
  const d = state.destination;
  const q = String(location == null ? '' : location).trim().slice(0, DEST_LOCATION_MAX);
  if (!q) {
    showDestHint(T().dest.emptyHint);
    $('dest-input').focus({ preventScroll: true });
    return;
  }
  hideDestHint();

  d.query = q;
  d.location = q;
  d.data = null;
  d.error = null;
  d.loading = true;
  d.seq += 1;
  const seq = d.seq;
  renderDestination();

  const minWait = new Promise((r) => setTimeout(r, DEST_MIN_WAIT_MS));

  const finish = (patch) => {
    if (seq !== d.seq) return;
    d.loading = false;
    Object.assign(d, patch);
    renderDestination();
    const title = $('dest-result-title');
    if (title) title.focus({ preventScroll: true });
  };

  try {
    const resp = await fetch('/api/destination', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location: q, language: state.lang }),
    });

    if (!resp.ok) {
      let detail = null;
      try {
        const body = await resp.json();
        if (typeof body.detail === 'string') detail = body.detail;
      } catch (_) { /* ignore parse failures */ }
      await minWait;
      finish({ error: { status: resp.status, detail } });
      return;
    }

    const data = await resp.json();
    await minWait;
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      finish({ error: { kind: 'badData' } });
      return;
    }
    finish({ data });
  } catch (_) {
    await minWait;
    finish({ error: { kind: 'network' } });
  }
}

/* ------------------------------------------- Travel context on the result page */

/**
 * The travel context card at the top of the result page: shown only if the user looked
 * up a destination during this session, and it must state that it **took no part in the
 * scores above** (the questionnaire has no location question).
 */
function renderTravelContext() {
  const t = T();
  const data = state.destination.data;
  const card = $('travel-context');
  const offer = $('travel-offer');

  // No destination looked up: offer only a small entry point
  if (!data) {
    card.hidden = true;
    $('travel-offer-text').textContent = t.travel.offer;
    $('btn-travel-offer').textContent = t.travel.offerLink;
    offer.hidden = false;
    return;
  }

  offer.hidden = true;
  const location = String(data.location == null ? '' : data.location).trim()
    || state.destination.location;

  card.classList.remove(...ENDEMICITY_LEVELS.map((l) => `endem-${l}`));
  const badge = $('travel-level');
  if (data.matched === true) {
    const level = ENDEMICITY_LEVELS.indexOf(data.endemicity) !== -1 ? data.endemicity : 'unknown';
    card.classList.add(`endem-${level}`);
    badge.textContent = fmt(t.travel.level, { level: t.dest.endemicity.levels[level] });
  } else {
    // This place is not in the reference table: do not guess a level
    card.classList.add('endem-unknown');
    badge.textContent = t.travel.notMatched;
  }

  $('travel-label').textContent = t.travel.label;
  $('travel-place').textContent = fmt(t.travel.place, { location });

  const season = String(data.season_note == null ? '' : data.season_note).trim();
  const seasonEl = $('travel-season');
  seasonEl.textContent = season;
  seasonEl.hidden = !season;

  card.hidden = false;
}

/* --------------------------------------------------------- "Why this score" */

// Metric <-> DOM mapping: the trigger button, the panel, and the I18N key of the metric name
const EXPLAIN_TARGETS = {
  dengue: { trigger: 'gauge-dengue', panel: 'explain-dengue', name: 'dengue' },
  worsening: { trigger: 'btn-explain-worsening', panel: 'explain-worsening', name: 'worsening' },
  severe: { trigger: 'gauge-severe', panel: 'explain-severe', name: 'severe' },
};

function explanationList(all, metric) {
  const list = all && all[metric];
  if (!Array.isArray(list)) return [];
  return list.filter((it) => it && typeof it === 'object');
}

function renderAllExplanations(all) {
  METRICS.forEach((metric) => {
    const target = EXPLAIN_TARGETS[metric];
    const items = explanationList(all, metric);
    const trigger = $(target.trigger);
    const panel = $(target.panel);

    // With no contributions, do not pretend it can be opened
    if (!items.length) {
      state.openExplain[metric] = false;
      trigger.disabled = true;
      trigger.setAttribute('aria-expanded', 'false');
      panel.hidden = true;
      panel.replaceChildren();
      return;
    }

    trigger.disabled = false;
    const open = !!state.openExplain[metric];
    trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
    panel.replaceChildren(buildExplainPanel(metric, items));
    panel.hidden = !open;
  });
}

function buildExplainPanel(metric, items) {
  const t = T();
  const ex = t.result.explain;
  const frag = document.createDocumentFragment();

  const title = document.createElement('p');
  title.className = 'explain-title';
  title.textContent = fmt(ex.title, { metric: t.result[metric] });
  frag.appendChild(title);

  const sub = document.createElement('p');
  sub.className = 'explain-sub';
  sub.textContent = `${ex.up} ↑ · ${ex.down} ↓`;
  frag.appendChild(sub);

  const max = items.reduce(
    (acc, it) => Math.max(acc, Math.abs(Number(it.contribution) || 0)),
    0,
  );

  const list = document.createElement('div');
  list.className = 'ex-list';
  let hasSeasonal = false;

  items.forEach((it) => {
    const value = Math.abs(Number(it.contribution) || 0);
    const dir = it.direction === 'down' ? 'down' : 'up';
    if (it.code === 'wk_sin' || it.code === 'wk_cos') hasSeasonal = true;

    const row = document.createElement('div');
    row.className = 'ex-row';

    const name = document.createElement('span');
    name.className = 'ex-name';
    name.textContent = featureLabel(it.code || it.feature);

    const bar = document.createElement('span');
    bar.className = 'ex-bar';
    const fill = document.createElement('span');
    fill.className = `ex-fill dir-${dir}`;
    // Width expresses relative size within this one list only; a 6% minimum keeps it visible
    const pct = max > 0 ? Math.max(6, Math.round((value / max) * 100)) : 6;
    fill.style.width = `${pct}%`;
    bar.appendChild(fill);

    const arrow = document.createElement('span');
    arrow.className = `ex-dir dir-${dir}`;
    const glyph = document.createElement('span');
    glyph.setAttribute('aria-hidden', 'true');
    glyph.textContent = dir === 'down' ? '↓' : '↑';
    const sr = document.createElement('span');
    sr.className = 'sr-only';
    sr.textContent = dir === 'down' ? ex.down : ex.up;
    arrow.append(glyph, sr);

    row.append(name, bar, arrow);
    list.appendChild(row);
  });

  frag.appendChild(list);


  
  return frag;
}

function toggleExplain(metric) {
  const target = EXPLAIN_TARGETS[metric];
  const trigger = $(target.trigger);
  if (trigger.disabled) return;
  const open = !state.openExplain[metric];
  state.openExplain[metric] = open;
  trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
  $(target.panel).hidden = !open;
}

/* --------------------------------------------------------- Errors */

function errorMessage() {
  const t = T().errors;
  if (!lastError) return t.server;
  if (lastError.kind === 'network') return t.network;
  if (lastError.kind === 'badData') return t.badData;
  if (lastError.detail) return lastError.detail;
  const s = lastError.status;
  if (s === 422) return t.validation;
  if (s === 502) return t.upstream;
  if (s >= 500) return t.server;
  return fmt(t.generic, { status: s });
}

function renderError() {
  $('error-detail').textContent = errorMessage();
}

function showErrorOverlay() {
  renderError();
  $('error-overlay').hidden = false;
}

function hideErrorOverlay() {
  $('error-overlay').hidden = true;
  lastError = null;
}

/* --------------------------------------------------------- Privacy overlay */

function renderPrivacy() {
  const t = T().privacy;
  $('privacy-title').textContent = t.title;
  $('privacy-intro').textContent = t.intro;
  $('privacy-list').replaceChildren(...t.items.map((text, i) => {
    const li = document.createElement('li');
    if (PRIVACY_KEY_ITEMS.indexOf(i) !== -1) li.className = 'is-key';
    li.textContent = text;
    return li;
  }));
  $('privacy-foot').textContent = t.foot;
  $('btn-privacy-close').textContent = t.close;
}

function openPrivacy() {
  renderPrivacy();
  lastFocusBeforeModal = document.activeElement;
  $('privacy-overlay').hidden = false;
  $('btn-privacy-close').focus();
}

function closePrivacy() {
  if ($('privacy-overlay').hidden) return;
  $('privacy-overlay').hidden = true;
  const back = lastFocusBeforeModal;
  lastFocusBeforeModal = null;
  if (back && typeof back.focus === 'function' && document.contains(back)) {
    back.focus();
  }
}

/* --------------------------------------------------------- Follow-up chat */

function chatContext() {
  const a = state.answers;
  const r = lastResult || {};
  const levels = ['low', 'medium', 'high'];
  const ctx = {
    warning_signs: Array.isArray(r.warning_signs) ? r.warning_signs : [],
    exposure_level:
      (r.exposure_context && levels.indexOf(r.exposure_context.level) !== -1)
        ? r.exposure_context.level
        : 'low',
    symptoms: { ...a.symptoms },
    comorbidities: { ...a.comorbidities },
    age: a.age,
    sex: a.sex,
    day_ill: a.dayIll,
  };
  METRICS.forEach((m) => {
    const s = r[m];
    if (s && typeof s === 'object' && levels.indexOf(s.level) !== -1) {
      ctx[m] = { score: Number(s.score) || 0, level: s.level };
    }
  });
  return ctx;
}

/** Send back the most recent few turns; the last of them is this very question, which
 *  goes in the question field and is not duplicated into history. */
function chatHistory() {
  const clean = state.chat.messages.filter((m) => !m.error && m.content);
  return clean
    .slice(0, -1)
    .slice(-CHAT_HISTORY_MAX)
    .map((m) => ({ role: m.role, content: m.content }));
}

function chatBubble(msg) {
  const t = T().chat;
  const el = document.createElement('div');
  const who = document.createElement('span');
  who.className = 'sr-only';

  if (msg.role === 'user') {
    el.className = 'chat-msg is-user';
    who.textContent = `${t.you}: `;
    el.append(who, document.createTextNode(msg.content));
    return el;
  }

  el.className = 'chat-msg is-assistant';
  who.textContent = `${t.assistant}: `;

  if (msg.error) {
    el.classList.add('is-error');
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'chat-retry';
    retry.textContent = t.retry;
    retry.disabled = state.chat.sending;
    retry.addEventListener('click', () => retryChat(msg));
    el.append(who, document.createTextNode(t.error), document.createElement('br'), retry);
    return el;
  }

  el.append(who, document.createTextNode(msg.content));

  // Sources carried by the answer: grouped into WHO official / web search, opening in a
  // new tab; render nothing at all when there are none
  const sources = buildSourcesBlock(msg.sources, 'msg-sources');
  if (sources) el.appendChild(sources);

  return el;
}

function typingBubble() {
  const el = document.createElement('div');
  el.className = 'chat-msg is-assistant chat-typing';
  const sr = document.createElement('span');
  sr.className = 'sr-only';
  sr.textContent = T().chat.typing;
  el.appendChild(sr);
  for (let i = 0; i < 3; i += 1) {
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.setAttribute('aria-hidden', 'true');
    el.appendChild(dot);
  }
  return el;
}

function renderChatLog() {
  const log = $('chat-log');
  const nodes = state.chat.messages.map(chatBubble);
  if (state.chat.sending) nodes.push(typingBubble());
  log.replaceChildren(...nodes);
  log.scrollTop = log.scrollHeight;
}

function renderChatChips() {
  const t = T().chat;
  const host = $('chat-chips');
  host.replaceChildren(...t.chips.map((text) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chat-chip';
    btn.textContent = text;
    btn.disabled = state.chat.sending;
    btn.addEventListener('click', () => askChat(text));
    return btn;
  }));
}

function updateChatCounter() {
  const input = $('chat-input');
  const n = input.value.length;
  const counter = $('chat-counter');
  counter.textContent = `${n} / ${CHAT_QUESTION_MAX}`;
  counter.classList.toggle('is-warn', n >= CHAT_QUESTION_MAX - 50);
  $('chat-send').disabled = state.chat.sending || !input.value.trim();
}

function autoGrowChatInput() {
  const el = $('chat-input');
  el.style.height = 'auto';
  // While the view is hidden scrollHeight is 0, so clear the inline height and hand
  // control back to the CSS min-height
  const h = el.scrollHeight;
  el.style.height = h > 0 ? `${Math.min(h, 132)}px` : '';
}

function setChatBusy(busy) {
  state.chat.sending = busy;
  $('chat-input').disabled = busy;
  document.querySelectorAll('.chat-chip').forEach((b) => { b.disabled = busy; });
  updateChatCounter();
}

function askChat(text) {
  const q = String(text || '').trim().slice(0, CHAT_QUESTION_MAX);
  if (!q || state.chat.sending) return;
  state.chat.messages.push({ role: 'user', content: q });
  runChatTurn(q);
}

function submitChatForm() {
  const input = $('chat-input');
  const text = input.value;
  if (!text.trim() || state.chat.sending) return;
  input.value = '';
  autoGrowChatInput();
  updateChatCounter();
  askChat(text);
}

function retryChat(errorMsg) {
  if (state.chat.sending || !errorMsg.question) return;
  const idx = state.chat.messages.indexOf(errorMsg);
  if (idx !== -1) state.chat.messages.splice(idx, 1);
  runChatTurn(errorMsg.question);
}

async function runChatTurn(question) {
  setChatBusy(true);
  renderChatLog();

  const body = {
    language: state.lang,
    question,
    context: chatContext(),
    history: chatHistory(),
  };

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const data = await resp.json();
    const reply = data && typeof data.reply === 'string' ? data.reply.trim() : '';
    if (!reply) throw new Error('empty reply');
    const sources = data && Array.isArray(data.sources) ? data.sources : [];
    state.chat.messages.push({ role: 'assistant', content: reply, sources });
  } catch (_) {
    // A chat failure is reported in the bubble only; no global error overlay
    state.chat.messages.push({ role: 'assistant', error: true, question });
  } finally {
    setChatBusy(false);
    renderChatLog();
  }
}

/* --------------------------------------------------------- Initialisation */

function resetWizard() {
  state.answers = freshAnswers();
  state.answered = freshAnsweredSet();
  state.adaptive = freshAdaptive();
  state.chat = freshChat();
  METRICS.forEach((m) => { state.openExplain[m] = false; });
  $('chat-input').value = '';
  autoGrowChatInput();
  updateChatCounter();
  renderChatLog();
  lastResult = null;
  state.step = 0;
  renderWizard(null);
}

/** Start a fresh assessment, remembering where its back control should return to. */
function openWizard(from) {
  state.wizardFrom = from === 'destination' ? 'destination' : 'hero';
  resetWizard();
  showView('wizard');
}

/** Leave the assessment entirely (the counterpart of the destination page's back
 *  control). btn-prev still steps backwards *inside* the wizard. */
function leaveWizard() {
  if (state.wizardFrom === 'destination') {
    renderDestination();
    showView('destination');
    return;
  }
  showView('hero');
}

function init() {
  state.mode = detectMode();
  updateModeToggle();
  setLanguage(detectLang());
  updateChatCounter();

  $('btn-start').addEventListener('click', () => openWizard('hero'));
  $('btn-wizard-back').addEventListener('click', leaveWizard);

  // ---- Destination lookup (the second path) ----
  $('btn-destination').addEventListener('click', () => openDestination('hero'));
  $('btn-dest-back').addEventListener('click', leaveDestination);

  $('dest-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runDestination($('dest-input').value);
  });

  $('dest-input').addEventListener('input', () => {
    state.destination.query = $('dest-input').value;
    hideDestHint();
  });

  $('dest-retry').addEventListener('click', () => {
    const d = state.destination;
    runDestination(d.location || d.query);
  });

  // Destination page -> symptom self-assessment: two distinct intents, kept explicitly
  // separate rather than sharing one flow
  $('btn-dest-to-symptoms').addEventListener('click', () => openWizard('destination'));

  // Result page -> destination lookup (the small entry point shown before any lookup)
  $('btn-travel-offer').addEventListener('click', () => openDestination('result'));

  // Interview mode switch: smart interview / full questionnaire (answers are shared, and
  // switching back to smart re-plans)
  $('mode-adaptive').addEventListener('click', () => {
    if (state.mode !== 'adaptive') setMode('adaptive');
  });
  $('mode-classic').addEventListener('click', () => {
    if (state.mode !== 'classic') setMode('classic');
  });

  $('btn-prev').addEventListener('click', () => {
    if (state.mode === 'adaptive') { adaptivePrev(); return; }
    if (state.step > 0) goStep(state.step - 1, 'back');
  });

  $('btn-next').addEventListener('click', () => {
    if (state.mode === 'adaptive') { adaptiveNext(); return; }
    if (!validateStep()) return;
    if (state.step === STEPS.length - 1) submit();
    else goStep(state.step + 1, 'forward');
  });

  // Smart mode: keys 1/2/3 answer the current question quickly
  document.addEventListener('keydown', (e) => {
    if (e.key !== '1' && e.key !== '2' && e.key !== '3') return;
    if ($('view-wizard').hidden || state.mode !== 'adaptive') return;
    const ad = state.adaptive;
    if (ad.stage !== 'loop' || !ad.current || ad.planning) return;
    if (!$('privacy-overlay').hidden || !$('error-overlay').hidden) return;
    const el = document.activeElement;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
    e.preventDefault();
    answerCurrent(e.key === '1' ? 'yes' : e.key === '2' ? 'no' : 'unknown');
  });

  // Restarting discards the previous result, so its back control goes home, not back
  // to a result page that no longer exists
  $('btn-restart').addEventListener('click', () => openWizard('hero'));

  $('btn-home').addEventListener('click', () => {
    lastResult = null;
    showView('hero');
  });

  $('btn-retry').addEventListener('click', () => {
    hideErrorOverlay();
    submit();
  });

  $('btn-error-close').addEventListener('click', () => {
    hideErrorOverlay();
    showView('wizard');
  });

  // Privacy overlay
  $('btn-privacy').addEventListener('click', openPrivacy);
  $('btn-privacy-chat').addEventListener('click', openPrivacy);
  $('btn-privacy-close').addEventListener('click', closePrivacy);
  $('privacy-overlay').querySelector('.overlay-backdrop')
    .addEventListener('click', closePrivacy);
  // Single-control dialog: Tab must not escape, and always lands on the close button
  $('privacy-overlay').addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      $('btn-privacy-close').focus();
    }
  });

  // "Why this score" -- the two gauges plus the worsening-risk bar
  METRICS.forEach((metric) => {
    $(EXPLAIN_TARGETS[metric].trigger)
      .addEventListener('click', () => toggleExplain(metric));
  });

  // Follow-up chat
  const chatInput = $('chat-input');
  chatInput.addEventListener('input', () => {
    updateChatCounter();
    autoGrowChatInput();
  });
  // Enter sends, Shift+Enter inserts a newline; an Enter during IME composition does not send
  chatInput.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    submitChatForm();
  });
  $('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    submitChatForm();
  });

  // Language switcher
  const toggle = $('lang-toggle');
  const menu = $('lang-menu');
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = menu.hidden;
    menu.hidden = !open;
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  document.querySelectorAll('.lang-option').forEach((btn) => {
    btn.addEventListener('click', () => {
      setLanguage(btn.dataset.lang);
      menu.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
  document.addEventListener('click', (e) => {
    if (!menu.hidden && !menu.contains(e.target) && e.target !== toggle) {
      menu.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!menu.hidden) {
      menu.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
    } else if (!$('privacy-overlay').hidden) {
      closePrivacy();
    } else if (!$('error-overlay').hidden) {
      hideErrorOverlay();
    }
  });

  showView('hero');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
