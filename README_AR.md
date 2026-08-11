# حزمة إعادة إنتاج بحث Cloud-Edge PPO

تحتوي هذه الحزمة على الكود والنتائج ونقاط حفظ PPO والرسوم المستخدمة لدعم بحث **Design and Evaluation of Adaptive PPO-Based Resource Orchestration for Latency-Sensitive AI Inference in Cloud-Edge Computing**.

الدراسة **محاكاة تحليلية حدثية**، ولا تدعي تشغيل ImageNet/CIFAR-10 أو قياس الطاقة على أجهزة فعلية. قيمة `accuracy` في الملفات هي **درجة احتفاظ تحليلية بالدقة** وليست دقة تصنيف مقاسة على dataset.

## التشغيل

```bash
pip install -r requirements.txt
python code/revised_framework.py
```

ولفحص تطابق الملفات المحفوظة:

```bash
python verify_repository.py
```

## محتويات الحزمة

- الكود الموحد ودفتر Jupyter وتسع خلايا منفصلة.
- إعدادات التجربة والبذور.
- النتائج الخام والملخصات.
- اختبارات Wilcoxon مع تصحيح Holm.
- نتائج Ablation.
- أربعة checkpoints لـ PPO.
- أربعة رسوم تجريبية.
- نسخة العمل الحالية من البحث للحفظ فقط.

## ملاحظة

يمكن رفع مجلدات الكود والنتائج والإعدادات والـcheckpoints والرسوم وملفات README إلى GitHub. بعد ذلك يفضل إنشاء إصدار Release وربطه بـ Zenodo للحصول على DOI ثابت قبل إدراج رابط المستودع في المخطوطة النهائية.
