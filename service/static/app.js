'use strict';

/* =========================================================
 * 登革热风险自测 — 前端逻辑
 *
 * 问卷字段与后端契约严格一致（app/schemas.py）：
 *   age(0-110) / sex(F|M) / day_ill(0-14)
 *   symptoms      —— 14 项，每项 yes | no | unknown
 *   comorbidities —— 7 项，每项 yes | no | unknown
 *   exposure      —— 3 项，每项 yes | no | unknown（**不参与模型打分**）
 *   language / notes
 *
 * 「不知道」在模型里与「无」同为 0（训练数据 SINAN 9=未知 记 0），
 * 所以未作答可以直接提交，不强制用户回答症状题。
 *
 * 流行病学暴露（exposure）在 SINAN 训练数据里不存在，没有系数可用，
 * 后端把它放进独立的规则通道 exposure_context，前端也必须区别呈现：
 * 它不是模型评分，不能长得像仪表盘。
 * ========================================================= */

const LANGS = [
  { code: 'zh-CN', name: '简体中文' },
  { code: 'zh-TW', name: '繁體中文' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'pt', name: 'Português' },
];

const DEFAULT_LANG = 'zh-CN';

// 与后端 SYMPTOM_CODES / COMORB_CODES 一致
const SYMPTOM_CODES = [
  'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS',
  'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N', 'LEUCOPENIA', 'LACO', 'DOR_RETRO',
];
const COMORB_CODES = [
  'DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE',
];
// 流行病学暴露：走后端的规则通道，不进模型
const EXPOSURE_CODES = ['FEVER_CLUSTER', 'CONFIRMED_CASE', 'OUTBREAK_TRAVEL'];

// 三态作答的字段类别（用于计数、批量填「无」等通用逻辑）
const TRI_KINDS = ['symptoms', 'comorbidities', 'exposure'];

// 模型贡献项里 5 个非二值特征（code 保留原名，不带 _x）
const NON_BINARY_FEATS = ['age', 'sex_f', 'day_ill', 'wk_sin', 'wk_cos'];

// 七步问卷结构（note 为 I18N 顶层键名）
const STEPS = [
  { id: 'basic', kind: 'basic' },
  { id: 'common', kind: 'symptoms', codes: ['FEBRE', 'CEFALEIA', 'MIALGIA', 'ARTRALGIA', 'DOR_RETRO', 'DOR_COSTAS'] },
  { id: 'other', kind: 'symptoms', codes: ['NAUSEA', 'VOMITO', 'EXANTEMA', 'CONJUNTVIT', 'ARTRITE'] },
  { id: 'clinical', kind: 'symptoms', codes: ['PETEQUIA_N', 'LACO', 'LEUCOPENIA'], note: 'clinicalNote' },
  { id: 'exposure', kind: 'exposure', codes: EXPOSURE_CODES, note: 'exposureNote' },
  { id: 'history', kind: 'comorbidities', codes: COMORB_CODES },
  { id: 'notes', kind: 'notes' },
];

// 结果页三个模型指标 ↔ explanations 的键
const METRICS = ['dengue', 'worsening', 'severe'];

// 建议展示顺序：就医 → 监测 → 防护（与后端 Advice 字段顺序一致）
const ADVICE_ORDER = ['medical', 'monitoring', 'protection'];

// 对话：最多回传最近 6 条历史，问题上限 500 字（与 app/schemas.py 一致）
const CHAT_HISTORY_MAX = 6;
const CHAT_QUESTION_MAX = 500;

/* --------------------------------------------------------- 文案 */

const I18N = {
  'zh-CN': {
    docTitle: '登革热风险自测',
    metaDescription: '回答症状与病史问题，获得登革热风险参考评分与防护建议。结果仅供参考，不构成医疗诊断。',
    langLabel: '选择语言',
    a11y: { progress: '问卷进度', chatLog: '对话记录' },
    hero: {
      badge: '登革热 · 风险自评',
      title: '登革热<br />风险自测',
      subtitle: '回答症状与病史问题，获得风险参考与防护建议',
      cta: '开始评估',
      privacy: '无需注册，不收集个人信息；补充说明不会被保存。',
      privacyLink: '隐私说明',
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
      basic: { title: '基本信息', sub: '这些信息会影响风险评估结果' },
      common: { title: '常见症状', sub: '最近是否出现以下情况？' },
      other: { title: '其他症状', sub: '继续，还有几项' },
      clinical: { title: '出血与化验', sub: '需要医生检查的项目' },
      exposure: { title: '周围环境与暴露', sub: '这几题问的是你身边的情况，不是你的身体' },
      history: { title: '既往病史', sub: '这些慢性病会提高重症风险' },
      notes: { title: '补充说明', sub: '还有什么想告诉我们的吗？（选填）' },
    },
    clinicalNote: '这几项通常需要医生检查或验血才能知道。不清楚就选「不知道」，不影响评估。',
    exposureNote: '这几题描述的是你周围的环境，而不是你的身体。统计模型并不使用它们（训练数据里没有这些变量），它们只用于判断流行病学背景、并影响给你的建议。',
    fields: {
      age: '年龄', ageUnit: '岁',
      sex: '性别', sexF: '女', sexM: '男',
      dayIll: '症状已持续', dayIllUnit: '天',
      dayIllZero: '今天刚开始', dayIllNone: '尚无症状',
    },
    answers: { yes: '有', no: '无', unknown: '不知道' },
    bulkNo: '本页全选「无」',
    answered: '已答 {n}/{total}',
    notes: {
      label: '补充说明',
      hint: '可以描述其他症状或情况，系统会自动识别其中提到的症状。最多 500 字。',
      placeholder: '例如：三天前开始发烧，眼睛后面很胀，昨天开始牙龈有点出血…',
    },
    nav: { prev: '上一步', next: '下一步', submit: '提交评估' },
    hints: { sexRequired: '请选择性别后继续' },
    loading: {
      steps: ['正在整理你的作答…', '模型评估中…', '正在生成个性化建议…'],
      sub: '请稍候，通常不超过 10 秒',
    },
    result: {
      title: '评估结果',
      sub: '基于你提供的症状与病史生成',
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
        caveat: '以上是模型对这个评分的线性贡献项：它解释分数是怎么算出来的，不代表病因。',
        seasonal: '「季节性周期」只取决于当前是第几周，与你本人无关：它今天让每个人的评分都平移同样的量，互相比较时会抵消掉。',
        empty: '这一项暂时没有可展示的贡献项。',
      },
      exposure: {
        label: '周围暴露情况',
        levels: { low: '低', medium: '中', high: '高' },
        factors: '触发因素：{list}',
        none: '你没有报告周围有相关的暴露情况。',
        caption: '这是按规则判断的流行病学背景，不是模型输出，也不计入上面的评分。',
        sep: '、',
      },
      epiWeek: '评估周次：第 {week} 周',
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
      sub: '针对上面的结果提问，AI 助手会用你选择的语言回答',
      empty: '先从下面挑一个问题，或者直接输入你想问的。',
      inputLabel: '你的问题',
      placeholder: '输入你的问题…',
      send: '发送',
      typing: 'AI 正在回复…',
      you: '你',
      assistant: 'AI 助手',
      note: '回答由 AI 生成，可能出错；你的作答摘要会发送给第三方模型接口处理。',
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
      badge: '登革熱 · 風險自評',
      title: '登革熱<br />風險自測',
      subtitle: '回答症狀與病史問題，獲得風險參考與防護建議',
      cta: '開始評估',
      privacy: '無需註冊，不蒐集個人資訊；補充說明不會被保存。',
      privacyLink: '隱私說明',
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
      basic: { title: '基本資料', sub: '這些資訊會影響風險評估結果' },
      common: { title: '常見症狀', sub: '最近是否出現以下情況？' },
      other: { title: '其他症狀', sub: '繼續，還有幾項' },
      clinical: { title: '出血與檢驗', sub: '需要醫師檢查的項目' },
      exposure: { title: '周遭環境與暴露', sub: '這幾題問的是你身邊的狀況，不是你的身體' },
      history: { title: '過去病史', sub: '這些慢性病會提高重症風險' },
      notes: { title: '補充說明', sub: '還有什麼想告訴我們的嗎？（選填）' },
    },
    clinicalNote: '這幾項通常需要醫師檢查或抽血才會知道。不清楚就選「不知道」，不影響評估。',
    exposureNote: '這幾題描述的是你周遭的環境，而不是你的身體。統計模型並不使用它們（訓練資料中沒有這些變項），它們只用來判斷流行病學背景，並影響給你的建議。',
    fields: {
      age: '年齡', ageUnit: '歲',
      sex: '性別', sexF: '女', sexM: '男',
      dayIll: '症狀已持續', dayIllUnit: '天',
      dayIllZero: '今天剛開始', dayIllNone: '尚無症狀',
    },
    answers: { yes: '有', no: '無', unknown: '不知道' },
    bulkNo: '本頁全選「無」',
    answered: '已答 {n}/{total}',
    notes: {
      label: '補充說明',
      hint: '可以描述其他症狀或狀況，系統會自動辨識其中提到的症狀。最多 500 字。',
      placeholder: '例如：三天前開始發燒，眼睛後面很脹，昨天開始牙齦有點出血…',
    },
    nav: { prev: '上一步', next: '下一步', submit: '送出評估' },
    hints: { sexRequired: '請先選擇性別再繼續' },
    loading: {
      steps: ['正在整理你的作答…', '模型評估中…', '正在產生個人化建議…'],
      sub: '請稍候，通常不超過 10 秒',
    },
    result: {
      title: '評估結果',
      sub: '根據你提供的症狀與病史產生',
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
        caveat: '以上是模型對這個評分的線性貢獻項：它解釋分數是怎麼算出來的，不代表病因。',
        seasonal: '「季節性週期」只取決於目前是第幾週，與你本人無關：它今天讓每個人的評分都平移相同的量，互相比較時會抵銷。',
        empty: '這一項目前沒有可顯示的貢獻項。',
      },
      exposure: {
        label: '周遭暴露情況',
        levels: { low: '低', medium: '中', high: '高' },
        factors: '觸發因素：{list}',
        none: '你沒有回報周遭有相關的暴露情況。',
        caption: '這是依規則判斷的流行病學背景，不是模型輸出，也不計入上方的評分。',
        sep: '、',
      },
      epiWeek: '評估週次：第 {week} 週',
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
      sub: '針對上面的結果提問，AI 助理會以你選擇的語言回答',
      empty: '先從下面挑一個問題，或直接輸入你想問的。',
      inputLabel: '你的問題',
      placeholder: '輸入你的問題…',
      send: '送出',
      typing: 'AI 正在回覆…',
      you: '你',
      assistant: 'AI 助理',
      note: '回答由 AI 產生，可能有誤；你的作答摘要會傳送至第三方模型 API 處理。',
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
      badge: 'Dengue · Risk self-check',
      title: 'Dengue Risk<br />Self-Check',
      subtitle: 'Answer a few questions to get a risk indicator and protection advice',
      cta: 'Start assessment',
      privacy: 'No account or personal details; free-text notes are never stored.',
      privacyLink: 'Privacy',
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
      basic: { title: 'Basic information', sub: 'These details affect the risk assessment' },
      common: { title: 'Common symptoms', sub: 'Have you had any of these recently?' },
      other: { title: 'Other symptoms', sub: 'Almost there — a few more' },
      clinical: { title: 'Bleeding & lab findings', sub: 'Items that usually require a clinician' },
      exposure: { title: 'Surroundings & exposure', sub: 'These ask about the people and places around you, not about your body' },
      history: { title: 'Medical history', sub: 'These conditions raise the risk of severe disease' },
      notes: { title: 'Anything else', sub: 'Optional — tell us more in your own words' },
    },
    clinicalNote: 'These usually require a doctor’s exam or a blood test. If you are not sure, choose "Don’t know" — it will not distort the result.',
    exposureNote: 'These questions describe your surroundings rather than your own body. The statistical model does not use them — the variables do not exist in its training data. They are used to judge the epidemiological context and to shape the guidance you receive.',
    fields: {
      age: 'Age', ageUnit: 'years',
      sex: 'Sex', sexF: 'Female', sexM: 'Male',
      dayIll: 'Symptoms have lasted', dayIllUnit: 'days',
      dayIllZero: 'Started today', dayIllNone: 'No symptoms yet',
    },
    answers: { yes: 'Yes', no: 'No', unknown: 'Don’t know' },
    bulkNo: 'Mark all on this page as "No"',
    answered: 'Answered {n}/{total}',
    notes: {
      label: 'Additional notes',
      hint: 'Describe any other symptoms or circumstances — the system will pick up symptoms you mention. Up to 500 characters.',
      placeholder: 'e.g. Fever started three days ago, pressure behind the eyes, gums began bleeding yesterday…',
    },
    nav: { prev: 'Back', next: 'Next', submit: 'Get my result' },
    hints: { sexRequired: 'Please select your sex to continue' },
    loading: {
      steps: ['Organising your answers…', 'Running the model…', 'Preparing your advice…'],
      sub: 'This usually takes less than 10 seconds',
    },
    result: {
      title: 'Your result',
      sub: 'Generated from the symptoms and history you provided',
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
        caveat: 'These are the model’s linear contributions to this score. They explain how the number was computed — they are not causes of your illness.',
        seasonal: 'The seasonal pattern depends only on the current week, not on you: today it shifts everyone’s score by the same amount, so it cancels out when scores are compared.',
        empty: 'No contributors are available for this score.',
      },
      exposure: {
        label: 'Surrounding exposure',
        levels: { low: 'Low', medium: 'Medium', high: 'High' },
        factors: 'Triggered by: {list}',
        none: 'You did not report any relevant exposure around you.',
        caption: 'Epidemiological context assessed by rule, not model output. It is not part of the scores above.',
        sep: ', ',
      },
      epiWeek: 'Assessment week: week {week}',
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
      sub: 'Ask anything about the result above — the AI assistant replies in your chosen language',
      empty: 'Pick a question below, or type your own.',
      inputLabel: 'Your question',
      placeholder: 'Type your question…',
      send: 'Send',
      typing: 'Assistant is replying…',
      you: 'You',
      assistant: 'Assistant',
      note: 'Replies are AI-generated and can be wrong. A summary of your answers is sent to a third-party model API.',
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
      badge: 'Dengue · Autoevaluación de riesgo',
      title: 'Riesgo de dengue<br />Autoevaluación',
      subtitle: 'Responda unas preguntas y obtenga un indicador de riesgo y consejos de protección',
      cta: 'Comenzar evaluación',
      privacy: 'Sin cuenta ni datos personales; los comentarios libres nunca se almacenan.',
      privacyLink: 'Privacidad',
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
      basic: { title: 'Datos básicos', sub: 'Estos datos influyen en la evaluación del riesgo' },
      common: { title: 'Síntomas frecuentes', sub: '¿Ha presentado alguno de estos últimamente?' },
      other: { title: 'Otros síntomas', sub: 'Continuemos, faltan pocos' },
      clinical: { title: 'Sangrado y laboratorio', sub: 'Requieren valoración médica' },
      exposure: { title: 'Entorno y exposición', sub: 'Estas preguntas son sobre lo que ocurre a su alrededor, no sobre su cuerpo' },
      history: { title: 'Antecedentes médicos', sub: 'Estas enfermedades aumentan el riesgo de gravedad' },
      notes: { title: 'Comentarios', sub: 'Opcional: cuéntenos algo más con sus palabras' },
    },
    clinicalNote: 'Estos datos suelen requerir revisión médica o análisis de sangre. Si no lo sabe, elija «No sé»: no afectará al resultado.',
    exposureNote: 'Estas preguntas describen su entorno, no su cuerpo. El modelo estadístico no las utiliza —esas variables no existen en sus datos de entrenamiento—: sirven para valorar el contexto epidemiológico y orientar las recomendaciones que recibe.',
    fields: {
      age: 'Edad', ageUnit: 'años',
      sex: 'Sexo', sexF: 'Mujer', sexM: 'Hombre',
      dayIll: 'Los síntomas llevan', dayIllUnit: 'días',
      dayIllZero: 'Comenzaron hoy', dayIllNone: 'Aún sin síntomas',
    },
    answers: { yes: 'Sí', no: 'No', unknown: 'No sé' },
    bulkNo: 'Marcar todo en esta página como «No»',
    answered: 'Respondidas {n}/{total}',
    notes: {
      label: 'Comentarios adicionales',
      hint: 'Describa otros síntomas o circunstancias; el sistema detectará los síntomas mencionados. Máximo 500 caracteres.',
      placeholder: 'Ej.: la fiebre empezó hace tres días, presión detrás de los ojos, ayer comenzó a sangrar la encía…',
    },
    nav: { prev: 'Atrás', next: 'Siguiente', submit: 'Ver mi resultado' },
    hints: { sexRequired: 'Seleccione su sexo para continuar' },
    loading: {
      steps: ['Organizando sus respuestas…', 'Ejecutando el modelo…', 'Preparando sus recomendaciones…'],
      sub: 'Normalmente tarda menos de 10 segundos',
    },
    result: {
      title: 'Su resultado',
      sub: 'Generado a partir de los síntomas y antecedentes indicados',
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
        caveat: 'Son las contribuciones lineales del modelo a esta puntuación: explican cómo se calculó la cifra, no son causas de su enfermedad.',
        seasonal: 'El patrón estacional depende solo de la semana del año, no de usted: hoy desplaza por igual la puntuación de todas las personas, así que se cancela al comparar.',
        empty: 'No hay factores disponibles para esta puntuación.',
      },
      exposure: {
        label: 'Exposición en su entorno',
        levels: { low: 'Baja', medium: 'Media', high: 'Alta' },
        factors: 'Motivos: {list}',
        none: 'No ha indicado ninguna exposición relevante en su entorno.',
        caption: 'Contexto epidemiológico determinado por reglas, no es resultado del modelo ni forma parte de las puntuaciones anteriores.',
        sep: ', ',
      },
      epiWeek: 'Semana de evaluación: semana {week}',
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
      sub: 'Pregunte lo que quiera sobre el resultado anterior; el asistente de IA responde en el idioma elegido',
      empty: 'Elija una pregunta de abajo o escriba la suya.',
      inputLabel: 'Su pregunta',
      placeholder: 'Escriba su pregunta…',
      send: 'Enviar',
      typing: 'El asistente está respondiendo…',
      you: 'Usted',
      assistant: 'Asistente',
      note: 'Las respuestas las genera una IA y pueden ser incorrectas. Se envía un resumen de sus respuestas a una API de modelo de terceros.',
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
      badge: 'Dengue · Autoavaliação de risco',
      title: 'Risco de dengue<br />Autoavaliação',
      subtitle: 'Responda a algumas perguntas e receba um indicador de risco e orientações',
      cta: 'Iniciar avaliação',
      privacy: 'Sem cadastro nem dados pessoais; as observações livres nunca são armazenadas.',
      privacyLink: 'Privacidade',
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
      basic: { title: 'Dados básicos', sub: 'Estes dados influenciam a avaliação de risco' },
      common: { title: 'Sintomas comuns', sub: 'Apresentou algum destes recentemente?' },
      other: { title: 'Outros sintomas', sub: 'Vamos continuar, faltam poucos' },
      clinical: { title: 'Sangramento e exames', sub: 'Itens que costumam exigir avaliação médica' },
      exposure: { title: 'Ambiente e exposição', sub: 'Estas perguntas são sobre o que acontece ao seu redor, não sobre o seu corpo' },
      history: { title: 'Histórico médico', sub: 'Estas condições aumentam o risco de gravidade' },
      notes: { title: 'Observações', sub: 'Opcional: conte-nos mais com suas palavras' },
    },
    clinicalNote: 'Estes itens geralmente exigem exame médico ou de sangue. Se não souber, escolha «Não sei» — isso não prejudica o resultado.',
    exposureNote: 'Estas perguntas descrevem o seu entorno, não o seu corpo. O modelo estatístico não as utiliza — essas variáveis não existem nos dados de treinamento. Elas servem para avaliar o contexto epidemiológico e orientar as recomendações que você recebe.',
    fields: {
      age: 'Idade', ageUnit: 'anos',
      sex: 'Sexo', sexF: 'Feminino', sexM: 'Masculino',
      dayIll: 'Os sintomas duram há', dayIllUnit: 'dias',
      dayIllZero: 'Começaram hoje', dayIllNone: 'Ainda sem sintomas',
    },
    answers: { yes: 'Sim', no: 'Não', unknown: 'Não sei' },
    bulkNo: 'Marcar tudo nesta página como «Não»',
    answered: 'Respondidas {n}/{total}',
    notes: {
      label: 'Observações adicionais',
      hint: 'Descreva outros sintomas ou circunstâncias; o sistema identificará os sintomas mencionados. Até 500 caracteres.',
      placeholder: 'Ex.: a febre começou há três dias, pressão atrás dos olhos, ontem a gengiva começou a sangrar…',
    },
    nav: { prev: 'Voltar', next: 'Avançar', submit: 'Ver meu resultado' },
    hints: { sexRequired: 'Selecione o sexo para continuar' },
    loading: {
      steps: ['Organizando suas respostas…', 'Executando o modelo…', 'Preparando suas orientações…'],
      sub: 'Normalmente leva menos de 10 segundos',
    },
    result: {
      title: 'Seu resultado',
      sub: 'Gerado a partir dos sintomas e do histórico informados',
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
        caveat: 'São as contribuições lineares do modelo para esta pontuação: explicam como o número foi calculado, não são causas da sua doença.',
        seasonal: 'O padrão sazonal depende apenas da semana do ano, não de você: hoje ele desloca igualmente a pontuação de todas as pessoas, portanto se cancela na comparação.',
        empty: 'Não há fatores disponíveis para esta pontuação.',
      },
      exposure: {
        label: 'Exposição ao seu redor',
        levels: { low: 'Baixa', medium: 'Média', high: 'Alta' },
        factors: 'Motivos: {list}',
        none: 'Você não relatou nenhuma exposição relevante ao seu redor.',
        caption: 'Contexto epidemiológico definido por regra, não é saída do modelo nem entra nas pontuações acima.',
        sep: ', ',
      },
      epiWeek: 'Semana da avaliação: semana {week}',
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
      sub: 'Pergunte o que quiser sobre o resultado acima; o assistente de IA responde no idioma escolhido',
      empty: 'Escolha uma pergunta abaixo ou escreva a sua.',
      inputLabel: 'Sua pergunta',
      placeholder: 'Escreva sua pergunta…',
      send: 'Enviar',
      typing: 'O assistente está respondendo…',
      you: 'Você',
      assistant: 'Assistente',
      note: 'As respostas são geradas por IA e podem estar erradas. Um resumo das suas respostas é enviado a uma API de modelo de terceiros.',
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

// 隐私条目里需要加粗的两条（各语言顺序一致）：不保存自由文本、AI 会外发摘要
const PRIVACY_KEY_ITEMS = [2, 5];

/* --------------------------------------------------------- 状态 */

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

const state = {
  lang: DEFAULT_LANG,
  step: 0,
  answers: freshAnswers(),
  submitting: false,
  // 三个贡献项面板的展开状态（切换语言后要保留）
  openExplain: { dengue: false, worsening: false, severe: false },
  chat: freshChat(),
};

let lastResult = null;
let lastError = null;
let loadingTimer = null;
let lastFocusBeforeModal = null;

/* --------------------------------------------------------- 工具 */

const $ = (id) => document.getElementById(id);
const T = () => I18N[state.lang];
const fmt = (tpl, vars) => tpl.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));

function itemText(kind, code) {
  const bag = T()[kind];
  return (bag && bag[code]) || { label: code, desc: '' };
}

/**
 * 把后端 explanations 里的 code 翻译成人话。
 * 症状 / 合并症去掉 _x 后查表；5 个非二值特征查 features 表。
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
  try { saved = localStorage.getItem('lang'); } catch (_) { /* 忽略 */ }
  if (saved && I18N[saved]) return saved;
  return mapNavLang(navigator.language);
}

function persistLang(code) {
  try { localStorage.setItem('lang', code); } catch (_) { /* 忽略 */ }
}

/* --------------------------------------------------------- 视图切换 */

const VIEWS = ['hero', 'wizard', 'loading', 'result'];

function showView(name) {
  VIEWS.forEach((v) => { $(`view-${v}`).hidden = v !== name; });
  if (name !== 'loading') stopLoadingRotation();
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

/* --------------------------------------------------------- 语言 */

function setLanguage(code) {
  if (!I18N[code]) code = DEFAULT_LANG;
  state.lang = code;
  persistLang(code);

  const t = T();
  document.documentElement.lang = code;
  document.title = t.docTitle;
  const meta = document.querySelector('meta[name="description"]');
  if (meta) meta.setAttribute('content', t.metaDescription);

  // 语言切换器
  const current = LANGS.find((l) => l.code === code);
  $('lang-current').textContent = current ? current.name : code;
  $('lang-toggle').setAttribute('aria-label', t.langLabel);
  $('lang-menu').setAttribute('aria-label', t.langLabel);
  document.querySelectorAll('.lang-option').forEach((btn) => {
    const on = btn.dataset.lang === code;
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
    btn.classList.toggle('is-current', on);
  });

  // Hero
  $('hero-badge').textContent = t.hero.badge;
  $('hero-title').innerHTML = t.hero.title;
  $('hero-subtitle').textContent = t.hero.subtitle;
  $('btn-start-text').textContent = t.hero.cta;
  $('hero-disclaimer').textContent = t.disclaimer;
  $('hero-privacy-text').textContent = t.hero.privacy;
  $('btn-privacy').textContent = t.hero.privacyLink;

  // Wizard 静态部分
  $('wizard-brand').textContent = t.brand;
  $('btn-prev').textContent = t.nav.prev;
  document.querySelector('.progress').setAttribute('aria-label', t.a11y.progress);

  // Loading
  $('loading-sub').textContent = t.loading.sub;

  // Result 静态部分
  $('result-title').textContent = t.result.title;
  $('result-sub').textContent = t.result.sub;
  $('label-dengue').textContent = t.result.dengue;
  $('label-severe').textContent = t.result.severe;
  $('label-worsening').textContent = t.result.worsening;
  $('why-dengue').textContent = t.result.explain.why;
  $('why-severe').textContent = t.result.explain.why;
  $('exposure-label').textContent = t.result.exposure.label;
  $('exposure-caption').textContent = t.result.exposure.caption;
  $('btn-restart').textContent = t.result.restart;
  $('btn-home').textContent = t.result.home;

  // 对话面板静态部分
  $('chat-title').textContent = t.chat.title;
  $('chat-sub').textContent = t.chat.sub;
  $('chat-empty').textContent = t.chat.empty;
  $('chat-log').setAttribute('aria-label', t.a11y.chatLog);
  $('chat-input-label').textContent = t.chat.inputLabel;
  $('chat-input').placeholder = t.chat.placeholder;
  $('chat-input').setAttribute('aria-label', t.chat.inputLabel);
  $('chat-send').textContent = t.chat.send;
  $('chat-note').textContent = t.chat.note;
  $('btn-privacy-chat').textContent = t.chat.privacyLink;

  // 错误浮层
  $('error-title').textContent = t.errors.title;
  $('btn-retry').textContent = t.errors.retry;
  $('btn-error-close').textContent = t.errors.back;

  // 隐私浮层
  renderPrivacy();

  // 免责声明（结果页优先用后端返回的）
  $('disclaimer-bar').textContent =
    (lastResult && lastResult.disclaimer) || t.disclaimer;

  // 重渲染动态内容
  if (!$('view-wizard').hidden) renderStep(null);
  if (!$('view-result').hidden && lastResult) renderResult(lastResult, { animate: false });
  if (!$('error-overlay').hidden && lastError) renderError();
  renderChatChips();
  renderChatLog();
}

/* --------------------------------------------------------- 问卷渲染 */

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
      opts.querySelectorAll('.q-opt').forEach((b) => {
        const sel = b.dataset.value === value;
        b.classList.toggle('is-on', sel);
        b.setAttribute('aria-pressed', sel ? 'true' : 'false');
      });
      updateStepMeta();
    });
    opts.appendChild(btn);
  });

  row.append(text, opts);
  return row;
}

function renderBasicStep(host) {
  const t = T();
  const a = state.answers;

  // 年龄
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

  // 性别
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

  // 病程天数
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

  const label = document.createElement('div');
  label.className = 'field-label';
  label.textContent = t.notes.label;
  const hint = document.createElement('p');
  hint.className = 'field-hint';
  hint.textContent = t.notes.hint;

  const ta = document.createElement('textarea');
  ta.className = 'notes-input';
  ta.maxLength = 500;
  ta.rows = 5;
  ta.placeholder = t.notes.placeholder;
  ta.value = state.answers.notes;
  ta.setAttribute('aria-label', t.notes.label);

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

  block.append(label, hint, ta, counter);
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

  // 标题
  const head = document.createElement('header');
  head.className = 'step-head';
  const h2 = document.createElement('h2');
  h2.className = 'step-title';
  h2.tabIndex = -1;
  h2.textContent = t.steps[step.id].title;
  const sub = document.createElement('p');
  sub.className = 'step-sub';
  sub.textContent = t.steps[step.id].sub;
  head.append(h2, sub);
  wrap.appendChild(head);

  // 步骤补充说明（临床项 / 流行病学暴露）
  if (step.note && t[step.note]) {
    const note = document.createElement('p');
    note.className = 'step-note';
    note.textContent = t[step.note];
    wrap.appendChild(note);
  }

  if (step.kind === 'basic') {
    renderBasicStep(wrap);
  } else if (step.kind === 'notes') {
    renderNotesStep(wrap);
  } else {
    // 批量「无」快捷键
    const bulk = document.createElement('button');
    bulk.type = 'button';
    bulk.className = 'bulk-btn';
    bulk.textContent = t.bulkNo;
    bulk.addEventListener('click', () => {
      step.codes.forEach((c) => { state.answers[step.kind][c] = 'no'; });
      renderStep(null);
    });
    wrap.appendChild(bulk);

    const list = document.createElement('div');
    list.className = 'q-list';
    step.codes.forEach((code) => list.appendChild(makeTriRow(step.kind, code)));
    wrap.appendChild(list);
  }

  panel.replaceChildren(wrap);
  updateStepMeta();
  hideHint();

  // 导航按钮
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

/* --------------------------------------------------------- 提交 */

function buildPayload() {
  const a = state.answers;
  return {
    age: a.age,
    sex: a.sex,
    day_ill: a.dayIll,
    symptoms: { ...a.symptoms },
    comorbidities: { ...a.comorbidities },
    // 不参与模型打分，后端用规则转成 exposure_context
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
      } catch (_) { /* 忽略解析失败 */ }
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
    // 新结果 = 新会话：清空追问记录与展开状态
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

/* --------------------------------------------------------- 结果渲染 */

const GAUGE_R = 92;
const GAUGE_C = 2 * Math.PI * GAUGE_R;

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function animateGauge(arcEl, scoreEl, target, animate) {
  const clamped = Math.max(0, Math.min(100, Number(target) || 0));
  const finalOffset = GAUGE_C * (1 - clamped / 100);
  arcEl.style.strokeDasharray = String(GAUGE_C);

  // 先落终值：即使动画帧不执行（后台标签页、节流环境），显示的也是正确结果
  arcEl.style.strokeDashoffset = String(finalOffset);
  scoreEl.textContent = clamped.toFixed(1);

  if (!animate || prefersReducedMotion()) {
    arcEl.style.transition = 'none';
    return;
  }

  // 圆环用 CSS 过渡驱动（不依赖 rAF）：归零 → 强制重排 → 设终值
  arcEl.style.transition = 'none';
  arcEl.style.strokeDashoffset = String(GAUGE_C);
  void arcEl.getBoundingClientRect();
  arcEl.style.transition = 'stroke-dashoffset 1.1s cubic-bezier(0.22, 1, 0.36, 1)';
  arcEl.style.strokeDashoffset = String(finalOffset);

  // 数字计数用 rAF；若不执行，上面已落的终值保持不变
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

  // 两个环形仪表盘
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

  // 加重风险横条
  const worse = data.worsening;
  const metric = $('metric-worsening');
  metric.classList.remove('level-low', 'level-medium', 'level-high');
  metric.classList.add(`level-${worse.level}`);
  $('score-worsening').textContent = Number(worse.score).toFixed(1);
  $('badge-worsening').textContent = t.result.levels[worse.level];
  const fill = $('bar-worsening');
  if (animate && !prefersReducedMotion()) {
    // 同样用「归零 → 强制重排 → 设终值」的 CSS 过渡，不依赖 rAF
    fill.style.transition = 'none';
    fill.style.width = '0%';
    void fill.getBoundingClientRect();
    fill.style.transition = '';
  }
  fill.style.width = `${worse.score}%`;

  // WHO 警示征象横幅（规则判断，独立于模型评分）
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

  // 文本
  $('result-summary').textContent = data.summary || '';
  $('model-note').textContent = data.model_note || '';
  $('epi-week-line').textContent = data.epi_week
    ? fmt(t.result.epiWeek, { week: data.epi_week })
    : '';
  $('disclaimer-bar').textContent = data.disclaimer || t.disclaimer;

  // 流行病学暴露背景（规则判断）
  renderExposure(data.exposure_context);

  // 模型贡献项面板（可展开）
  renderAllExplanations(data.explanations);

  // 建议卡片：就医 → 监测 → 防护
  const host = $('advice-cards');
  host.replaceChildren();
  ADVICE_ORDER.forEach((key, idx) => {
    const items = (data.advice && data.advice[key]) || [];
    if (!items.length) return;
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

  // 追问对话
  renderChatChips();
  renderChatLog();
}

/* --------------------------------------------------------- 暴露背景 */

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
  $('exposure-caption').textContent = t.caption;
  chip.hidden = false;
}

/* --------------------------------------------------------- 「为什么是这个分数」 */

// 指标 ↔ DOM 的映射：触发按钮、面板、指标名的 I18N 键
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

    // 没有贡献项就别假装可以点开
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
    // 宽度只表示同一列表内的相对大小，最小 6% 保证可见
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

  const caveat = document.createElement('p');
  caveat.className = 'explain-note';
  caveat.textContent = ex.caveat;
  frag.appendChild(caveat);

  if (hasSeasonal) {
    const seasonal = document.createElement('p');
    seasonal.className = 'explain-note';
    seasonal.textContent = ex.seasonal;
    frag.appendChild(seasonal);
  }

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

/* --------------------------------------------------------- 错误 */

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

/* --------------------------------------------------------- 隐私说明浮层 */

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

/* --------------------------------------------------------- 追问对话 */

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

/** 回传最近若干轮；末尾那条正是本次提问，交给 question 字段，不重复放进 history。 */
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
  const t = T().chat;
  const log = $('chat-log');
  const empty = $('chat-empty');
  empty.textContent = t.empty;
  empty.hidden = state.chat.messages.length > 0 || state.chat.sending;

  const nodes = [empty];
  state.chat.messages.forEach((m) => nodes.push(chatBubble(m)));
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
  // 视图隐藏时 scrollHeight 为 0，此时清空内联高度，交回 CSS 的 min-height
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
    state.chat.messages.push({ role: 'assistant', content: reply });
  } catch (_) {
    // 对话失败只在气泡里提示，不弹全局错误浮层
    state.chat.messages.push({ role: 'assistant', error: true, question });
  } finally {
    setChatBusy(false);
    renderChatLog();
  }
}

/* --------------------------------------------------------- 初始化 */

function resetWizard() {
  state.answers = freshAnswers();
  state.chat = freshChat();
  METRICS.forEach((m) => { state.openExplain[m] = false; });
  $('chat-input').value = '';
  autoGrowChatInput();
  updateChatCounter();
  renderChatLog();
  lastResult = null;
  goStep(0, null);
}

function init() {
  setLanguage(detectLang());
  updateChatCounter();

  $('btn-start').addEventListener('click', () => {
    resetWizard();
    showView('wizard');
  });

  $('btn-prev').addEventListener('click', () => {
    if (state.step > 0) goStep(state.step - 1, 'back');
  });

  $('btn-next').addEventListener('click', () => {
    if (!validateStep()) return;
    if (state.step === STEPS.length - 1) submit();
    else goStep(state.step + 1, 'forward');
  });

  $('btn-restart').addEventListener('click', () => {
    resetWizard();
    showView('wizard');
  });

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

  // 隐私说明浮层
  $('btn-privacy').addEventListener('click', openPrivacy);
  $('btn-privacy-chat').addEventListener('click', openPrivacy);
  $('btn-privacy-close').addEventListener('click', closePrivacy);
  $('privacy-overlay').querySelector('.overlay-backdrop')
    .addEventListener('click', closePrivacy);
  // 单控件对话框：Tab 不外逃，始终停在关闭按钮上
  $('privacy-overlay').addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      $('btn-privacy-close').focus();
    }
  });

  // 「为什么是这个分数」——两个仪表盘 + 加重风险横条
  METRICS.forEach((metric) => {
    $(EXPLAIN_TARGETS[metric].trigger)
      .addEventListener('click', () => toggleExplain(metric));
  });

  // 追问对话
  const chatInput = $('chat-input');
  chatInput.addEventListener('input', () => {
    updateChatCounter();
    autoGrowChatInput();
  });
  // Enter 送出、Shift+Enter 换行；输入法组字中的 Enter 不算送出
  chatInput.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    submitChatForm();
  });
  $('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    submitChatForm();
  });

  // 语言切换器
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
