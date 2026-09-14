# Bank Customer Churn Intelligence System

هذا المشروع يقدم لوحة Streamlit ثنائية اللغة (العربية/English) لتقدير مخاطر مغادرة عملاء البنك، مع تحليل فردي وجماعي ورسوم ومؤشرات أداء.

## التشغيل داخل VS Code

1. افتح مجلد `Dr_Muhannad_AI_Project` في VS Code.
2. افتح `Churn_Project_Final.ipynb`.
3. اختر Python Kernel.
4. استخدم **Run All** لتشغيل الخلايا بالترتيب.
5. شغّل الواجهة من Terminal داخل مجلد المشروع:

```powershell
py -m streamlit run app.py
```

ثم افتح الرابط الذي يظهر، غالبًا `http://localhost:8501`.

### تشغيل الواجهة بنقرة واحدة

بعد تثبيت المتطلبات، اضغط مرتين على الملف:

`تشغيل_الواجهة.bat`

سيتم تشغيل الواجهة وفتحها على `http://localhost:8501`. إذا لم تكن المتطلبات
مثبتة، شغّل الأمر التالي مرة واحدة من Terminal داخل مجلد المشروع:

```powershell
py -m pip install -r requirements.txt
```

## محتويات المشروع

| الملف | الوظيفة |
|---|---|
| `Churn_Project_Final.ipynb` | المشروع كاملًا خلية بخليه: اكتشاف، تقسيم، معالجة، تدريب، تقييم وتحليل. |
| `data/Churn_Modelling.csv` | البيانات الأصلية. |
| `best_model.joblib` | النموذج الأفضل المحفوظ بعد التدريب. |
| `app.py` | واجهة التنبؤ. |
| `src/opportunity.py` | Opportunity Engine: explainable scoring and demo campaign arithmetic. |
| `train_final.py` | إعادة تدريب النموذج وتحديث ملفه. |
| `outputs/metrics.json` | نتائج المقارنة. |

## التدريب والتقييم

```powershell
py train_final.py
```

يستخدم التدريب تقسيمًا طبقيًا إلى تدريب/تحقق/اختبار، و`class_weight="balanced"` لمعالجة عدم توازن المغادرة. تُختار عتبة القرار التي تزيد F1 على مجموعة التحقق فقط (ولا تُستخدم مجموعة الاختبار في الاختيار)، ثم يُعاد تدريب النموذج الأفضل على بيانات التدريب. تُحفظ العتبة والمنهج ومقاييس الاختبار في `outputs/metrics.json` وتستخدمها الواجهة تلقائيًا.

## ملاحظة مهمة

ملف `best_model.joblib` ملف ثنائي، لذلك لا يُفتح كنص. إذا ظهر في الواجهة احتمال تنبؤ، فهذا يعني أنه محفوظ ويعمل.

## Bank AI Early Warning & Retention Platform

The Streamlit app presents one clear operating flow:
**bank customer data → AI churn probability → 0–100 risk score → retention
action**. The primary navigation is intentionally limited to five sections:

- **Executive Overview**: demo flow, portfolio KPIs, and the main sync CTA.
- **Bank Sync**: run the safe in-process mock sync and inspect its result.
- **Early Warning**: filter the prioritized queue by risk band and score.
- **Customer Profile**: inspect a customer’s probability, drivers, action, and
  what-if scenario.
- **Advanced Tools**: the existing Individual, Batch, Data quality, Model
  performance, Analytics, Model Arena, Explainable AI, audit, settings, and
  integration pages remain available in the secondary selector.

The interface is presentation-ready in both light and dark Streamlit themes.
Demo Mode uses sample records only; it does not make real-bank claims or
require credentials.

## Innovation: Bank AI Retention Opportunity Engine

The primary product is now the **Retention Command Center**, not a churn
dashboard. It turns the persisted churn signal into an explainable opportunity
score:

**opportunity = churn risk × demo relationship-value proxy × intervention
opportunity**.

The weights are configurable in the UI and every score exposes its three
components, priority tier (P1–P4), segment, and recommended playbook. The
command center includes a top-10 next-action queue, segment opportunity chart,
and a campaign simulator. The simulator accepts an intervention cost and an
expected save-rate assumption and reports scenario arithmetic only. “Estimated
portfolio value at risk” uses the demo dataset's churn probability multiplied
by a clearly labelled balance/salary relationship-value proxy; it is **not**
real bank revenue, a financial forecast, or a claim about a financial
institution.

### Academic demo pitch

“A conventional model tells a bank who may leave. Bank AI Retention
Opportunity Engine answers the harder operating question: where should a team
start, why there, and what could an assumed intervention look like? In one
screen, an audience can change the priorities, inspect the transparent
trade-offs, choose a playbook, and test a hypothetical campaign—without
changing the underlying ML result or implying access to real bank data.”

Opportunity fields are additive to predictor responses and batch outputs; the
original probability, threshold, risk bands, reports, and analyst tools remain
available unchanged.

### Academic Demo Mode Bank AI Integration Center

تتضمن الواجهة الآن طبقة تكامل مستقلة وقابلة للاختبار داخل `src/` مع Demo Mode
آمن افتراضياً. لا يحتاج Demo Mode إلى مفاتيح أو خدمة خارجية، ويستخدم
`MockBankClient` داخل العملية. تُحفظ العملاء والتنبؤات وعمليات المزامنة وسجل
التدقيق في SQLite (`outputs/bank_ai.sqlite3`). انسخ `.env.example` إلى `.env`
لتغيير المسارات محلياً؛ لا تضع أسراراً في المستودع.

تظل أدوات AI Prediction وBank Integration وSynchronization وWhat-if وAnalytics
وModel Arena وExplainable AI وSecurity & Audit وSettings/Demo Mode، إضافة إلى
صفحات Individual وBatch وData quality وتقارير CSV/PDF، متاحة من Advanced Tools.
مستويات الخطر قابلة للتعديل:
0–24 LOW، 25–49 MEDIUM، 50–74 HIGH، 75–100 CRITICAL.

### API الاختياري

إذا كانت FastAPI مثبتة، يمكن تشغيل محول الـAPI من Python:

```python
from src.mock_api import create_app
app = create_app()
```

يوفر `/api/customers` و`/api/customers/{id}` و`POST /api/customers` و`POST
/api/sync` و`/api/health` و`/api/statistics` و`POST /api/webhook`. عدم تثبيت
FastAPI لا يؤثر على Streamlit Demo Mode.

### الاختبارات والتحقق

```powershell
py -m unittest discover -s tests -v
py -m py_compile app.py train_final.py src\*.py tests\*.py
```

يعرض Model Arena المقاييس الموجودة فعلياً في `outputs/metrics.json` دون اختلاق
مؤشرات. النموذج محفوظ ومحمّل مرة واحدة لكل مسار، وتعرض Explainable AI أهمية
الخصائص المستخرجة من pipeline.

## المنهج

المشروع يتبع الترتيب: **Data Discovery → Stratified Train/Validation/Test Split → Preprocessing → Imbalance-aware Model Training → Validation Threshold Selection → Evaluation → Business Analysis → Prediction**. تمت مقارنة Logistic Regression وDecision Tree وRandom Forest، واختيار النموذج الأفضل حسب F1-score على الاختبار باستخدام عتبة اختيرت مسبقًا من التحقق.
