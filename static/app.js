'use strict';

/* =========================================================
 * 登革热风险自测 — 前端逻辑
 *
 * 问卷字段与后端契约严格一致（app/schemas.py）：
 *   age(0-110) / sex(F|M) / day_ill(0-14)
 *   symptoms      —— 14 项，每项 yes | no | unknown
 *   comorbidities —— 7 项，每项 yes | no | unknown
 *   language / notes
 *
 * 「不知道」在模型里与「无」同为 0（训练数据 SINAN 9=未知 记 0），
 * 所以未作答可以直接提交，不强制用户回答症状题。
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

// 六步问卷结构
const STEPS = [
  { id: 'basic', kind: 'basic' },
  { id: 'common', kind: 'symptoms', codes: ['FEBRE', 'CEFALEIA', 'MIALGIA', 'ARTRALGIA', 'DOR_RETRO', 'DOR_COSTAS'] },
  { id: 'other', kind: 'symptoms', codes: ['NAUSEA', 'VOMITO', 'EXANTEMA', 'CONJUNTVIT', 'ARTRITE'] },
  { id: 'clinical', kind: 'symptoms', codes: ['PETEQUIA_N', 'LACO', 'LEUCOPENIA'], note: true },
  { id: 'history', kind: 'comorbidities', codes: COMORB_CODES },
  { id: 'notes', kind: 'notes' },
];

/* --------------------------------------------------------- 文案 */

const I18N = {
  'zh-CN': {
    docTitle: '登革热风险自测',
    metaDescription: '回答症状与病史问题，获得登革热风险参考评分与防护建议。结果仅供参考，不构成医疗诊断。',
    langLabel: '选择语言',
    a11y: { points: '产品特色', progress: '问卷进度' },
    hero: {
      badge: '登革热 · 风险自评',
      title: '登革热<br />风险自测',
      subtitle: '回答症状与病史问题，获得风险参考与防护建议',
      cta: '开始评估',
      points: ['🦟 登革热专用', '📊 真实数据建模', '🌏 五种语言'],
    },
    brand: '🦟 登革热风险自测',
    stepCounter: '第 {cur} 步 / 共 {total} 步',
    steps: {
      basic: { title: '基本信息', sub: '这些信息会影响风险评估结果' },
      common: { title: '常见症状', sub: '最近是否出现以下情况？' },
      other: { title: '其他症状', sub: '继续，还有几项' },
      clinical: { title: '出血与化验', sub: '需要医生检查的项目' },
      history: { title: '既往病史', sub: '这些慢性病会提高重症风险' },
      notes: { title: '补充说明', sub: '还有什么想告诉我们的吗？（选填）' },
    },
    clinicalNote: '这几项通常需要医生检查或验血才能知道。不清楚就选「不知道」，不影响评估。',
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
      advice: { protection: '🛡️ 防蚊与防护', medical: '🏥 就医提示', monitoring: '📋 居家监测' },
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
  },

  'zh-TW': {
    docTitle: '登革熱風險自測',
    metaDescription: '回答症狀與病史問題，獲得登革熱風險參考評分與防護建議。結果僅供參考，不構成醫療診斷。',
    langLabel: '選擇語言',
    a11y: { points: '產品特色', progress: '問卷進度' },
    hero: {
      badge: '登革熱 · 風險自評',
      title: '登革熱<br />風險自測',
      subtitle: '回答症狀與病史問題，獲得風險參考與防護建議',
      cta: '開始評估',
      points: ['🦟 登革熱專用', '📊 真實資料建模', '🌏 五種語言'],
    },
    brand: '🦟 登革熱風險自測',
    stepCounter: '第 {cur} 步 / 共 {total} 步',
    steps: {
      basic: { title: '基本資料', sub: '這些資訊會影響風險評估結果' },
      common: { title: '常見症狀', sub: '最近是否出現以下情況？' },
      other: { title: '其他症狀', sub: '繼續，還有幾項' },
      clinical: { title: '出血與檢驗', sub: '需要醫師檢查的項目' },
      history: { title: '過去病史', sub: '這些慢性病會提高重症風險' },
      notes: { title: '補充說明', sub: '還有什麼想告訴我們的嗎？（選填）' },
    },
    clinicalNote: '這幾項通常需要醫師檢查或抽血才會知道。不清楚就選「不知道」，不影響評估。',
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
      advice: { protection: '🛡️ 防蚊與防護', medical: '🏥 就醫提示', monitoring: '📋 居家監測' },
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
  },

  en: {
    docTitle: 'Dengue Risk Self-Check',
    metaDescription: 'Answer questions about symptoms and medical history to get a dengue risk indicator and protection advice. For reference only, not a medical diagnosis.',
    langLabel: 'Select language',
    a11y: { points: 'Highlights', progress: 'Questionnaire progress' },
    hero: {
      badge: 'Dengue · Risk self-check',
      title: 'Dengue Risk<br />Self-Check',
      subtitle: 'Answer a few questions to get a risk indicator and protection advice',
      cta: 'Start assessment',
      points: ['🦟 Dengue-specific', '📊 Built on real data', '🌏 Five languages'],
    },
    brand: '🦟 Dengue Risk Self-Check',
    stepCounter: 'Step {cur} of {total}',
    steps: {
      basic: { title: 'Basic information', sub: 'These details affect the risk assessment' },
      common: { title: 'Common symptoms', sub: 'Have you had any of these recently?' },
      other: { title: 'Other symptoms', sub: 'Almost there — a few more' },
      clinical: { title: 'Bleeding & lab findings', sub: 'Items that usually require a clinician' },
      history: { title: 'Medical history', sub: 'These conditions raise the risk of severe disease' },
      notes: { title: 'Anything else', sub: 'Optional — tell us more in your own words' },
    },
    clinicalNote: 'These usually require a doctor’s exam or a blood test. If you are not sure, choose "Don’t know" — it will not distort the result.',
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
      advice: { protection: '🛡️ Mosquito protection', medical: '🏥 When to seek care', monitoring: '📋 Monitoring at home' },
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
  },

  es: {
    docTitle: 'Autoevaluación de riesgo de dengue',
    metaDescription: 'Responda preguntas sobre síntomas y antecedentes para obtener un indicador de riesgo de dengue y recomendaciones de protección. Solo orientativo, no es un diagnóstico médico.',
    langLabel: 'Seleccionar idioma',
    a11y: { points: 'Características', progress: 'Progreso del cuestionario' },
    hero: {
      badge: 'Dengue · Autoevaluación de riesgo',
      title: 'Riesgo de dengue<br />Autoevaluación',
      subtitle: 'Responda unas preguntas y obtenga un indicador de riesgo y consejos de protección',
      cta: 'Comenzar evaluación',
      points: ['🦟 Específico para dengue', '📊 Basado en datos reales', '🌏 Cinco idiomas'],
    },
    brand: '🦟 Autoevaluación de dengue',
    stepCounter: 'Paso {cur} de {total}',
    steps: {
      basic: { title: 'Datos básicos', sub: 'Estos datos influyen en la evaluación del riesgo' },
      common: { title: 'Síntomas frecuentes', sub: '¿Ha presentado alguno de estos últimamente?' },
      other: { title: 'Otros síntomas', sub: 'Continuemos, faltan pocos' },
      clinical: { title: 'Sangrado y laboratorio', sub: 'Requieren valoración médica' },
      history: { title: 'Antecedentes médicos', sub: 'Estas enfermedades aumentan el riesgo de gravedad' },
      notes: { title: 'Comentarios', sub: 'Opcional: cuéntenos algo más con sus palabras' },
    },
    clinicalNote: 'Estos datos suelen requerir revisión médica o análisis de sangre. Si no lo sabe, elija «No sé»: no afectará al resultado.',
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
      advice: { protection: '🛡️ Protección contra mosquitos', medical: '🏥 Cuándo buscar atención', monitoring: '📋 Vigilancia en casa' },
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
  },

  pt: {
    docTitle: 'Autoavaliação de risco de dengue',
    metaDescription: 'Responda a perguntas sobre sintomas e histórico para obter um indicador de risco de dengue e orientações de proteção. Apenas para referência, não é um diagnóstico médico.',
    langLabel: 'Selecionar idioma',
    a11y: { points: 'Destaques', progress: 'Progresso do questionário' },
    hero: {
      badge: 'Dengue · Autoavaliação de risco',
      title: 'Risco de dengue<br />Autoavaliação',
      subtitle: 'Responda a algumas perguntas e receba um indicador de risco e orientações',
      cta: 'Iniciar avaliação',
      points: ['🦟 Específico para dengue', '📊 Baseado em dados reais', '🌏 Cinco idiomas'],
    },
    brand: '🦟 Autoavaliação de dengue',
    stepCounter: 'Etapa {cur} de {total}',
    steps: {
      basic: { title: 'Dados básicos', sub: 'Estes dados influenciam a avaliação de risco' },
      common: { title: 'Sintomas comuns', sub: 'Apresentou algum destes recentemente?' },
      other: { title: 'Outros sintomas', sub: 'Vamos continuar, faltam poucos' },
      clinical: { title: 'Sangramento e exames', sub: 'Itens que costumam exigir avaliação médica' },
      history: { title: 'Histórico médico', sub: 'Estas condições aumentam o risco de gravidade' },
      notes: { title: 'Observações', sub: 'Opcional: conte-nos mais com suas palavras' },
    },
    clinicalNote: 'Estes itens geralmente exigem exame médico ou de sangue. Se não souber, escolha «Não sei» — isso não prejudica o resultado.',
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
      advice: { protection: '🛡️ Proteção contra mosquitos', medical: '🏥 Quando procurar atendimento', monitoring: '📋 Monitoramento em casa' },
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
  },
};

/* --------------------------------------------------------- 状态 */

function freshAnswers() {
  const symptoms = {};
  SYMPTOM_CODES.forEach((c) => { symptoms[c] = 'unknown'; });
  const comorbidities = {};
  COMORB_CODES.forEach((c) => { comorbidities[c] = 'unknown'; });
  return { symptoms, comorbidities, age: 30, sex: null, dayIll: 2, notes: '' };
}

const state = {
  lang: DEFAULT_LANG,
  step: 0,
  answers: freshAnswers(),
  submitting: false,
};

let lastResult = null;
let lastError = null;
let loadingTimer = null;

/* --------------------------------------------------------- 工具 */

const $ = (id) => document.getElementById(id);
const T = () => I18N[state.lang];
const fmt = (tpl, vars) => tpl.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));

function itemText(kind, code) {
  return T()[kind][code] || { label: code, desc: '' };
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
  const points = $('hero-points');
  points.setAttribute('aria-label', t.a11y.points);
  points.replaceChildren(...t.hero.points.map((p) => {
    const li = document.createElement('li');
    li.textContent = p;
    return li;
  }));
  $('hero-disclaimer').textContent = t.disclaimer;

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
  $('btn-restart').textContent = t.result.restart;
  $('btn-home').textContent = t.result.home;

  // 错误浮层
  $('error-title').textContent = t.errors.title;
  $('btn-retry').textContent = t.errors.retry;
  $('btn-error-close').textContent = t.errors.back;

  // 免责声明（结果页优先用后端返回的）
  $('disclaimer-bar').textContent =
    (lastResult && lastResult.disclaimer) || t.disclaimer;

  // 重渲染动态内容
  if (!$('view-wizard').hidden) renderStep(null);
  if (!$('view-result').hidden && lastResult) renderResult(lastResult, { animate: false });
  if (!$('error-overlay').hidden && lastError) renderError();
}

/* --------------------------------------------------------- 问卷渲染 */

function answeredCount(step) {
  if (step.kind !== 'symptoms' && step.kind !== 'comorbidities') return null;
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

  // 临床项提示
  if (step.note) {
    const note = document.createElement('p');
    note.className = 'step-note';
    note.textContent = t.clinicalNote;
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

  // 建议卡片
  const host = $('advice-cards');
  host.replaceChildren();
  ['protection', 'medical', 'monitoring'].forEach((key, idx) => {
    const items = (data.advice && data.advice[key]) || [];
    if (!items.length) return;
    const card = document.createElement('section');
    card.className = 'card advice-card';
    if (animate && !prefersReducedMotion()) {
      card.classList.add('fade-up');
      card.style.animationDelay = `${0.12 * idx}s`;
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

/* --------------------------------------------------------- 初始化 */

function resetWizard() {
  state.answers = freshAnswers();
  lastResult = null;
  goStep(0, null);
}

function init() {
  setLanguage(detectLang());

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
