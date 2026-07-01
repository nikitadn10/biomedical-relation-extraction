# Biomedical Drug-Disease Relation Extraction

Fine-tuned **BioBERT** for 3-class relation extraction from clinical text, combining transformer-based classification with spaCy dependency parsing on the BC5CDR benchmark corpus.

---

## What It Does

Given a clinical sentence, the pipeline extracts structured triplets of the form:

```
(Drug) --[RELATION]--> (Disease)
```

where RELATION is one of:
- **TREATS** — drug is used to treat the disease
- **CAUSES** — drug causes the disease as a side effect
- **NO_RELATION** — entities co-occur but have no direct relation

### Example Output

| Drug | Symptom | Relation | Method | Dependency Path |
|------|---------|----------|--------|-----------------|
| Aspirin | headache | TREATS | Dependency rule | Aspirin → relieved → headache |
| Penicillin | rash | CAUSES | Dependency rule | Penicillin → produced → rash |
| Amoxicillin | bacterial infection | TREATS | Dependency rule | Amoxicillin → treated → infection |
| Ibuprofen | vomiting | CAUSES | Dependency rule | Ibuprofen → caused → vomiting |

---

## Results

Evaluated on BC5CDR validation set (403 samples):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| TREATS | 0.853 | 0.825 | **0.839** |
| CAUSES | 0.652 | 0.698 | **0.674** |
| **Macro F1** | 0.753 | 0.761 | **0.756** |
| **Weighted F1** | 0.789 | 0.784 | **0.786** |

---

## Dataset

**BC5CDR** — BioCreative V Chemical-Disease Relation Corpus (NCBI)
- 1,500 annotated PubMed abstracts
- 1,978 balanced training samples (CAUSES: 542, TREATS: 1068, NO_RELATION: 368)
- Citation: Li et al. (2016). *BioCreative V CDR task corpus*. Database, 2016.

---

## Architecture

```
Clinical Sentence
      ↓
BC5CDR NER (Chemical + Disease entities)
      ↓
spaCy Dependency Parser → Shortest Path between entities
      ↓
Rule-based check (treat/cause verbs) → if matched, assign directly
      ↓
BioBERT Classifier (fallback) → [DRUG] ... [/DRUG] entity markers
      ↓
(Drug, Disease, Relation) Triplet
```

**Two-stage classification:**
1. Dependency rules take priority — if the shortest path contains verbs like *treat, prescribe, reduce, cause, induce*, the relation is assigned directly
2. Fine-tuned BioBERT acts as fallback for ambiguous cases

---

## Tech Stack

- `dmis-lab/biobert-base-cased-v1.2` — domain-specific BERT for biomedical text
- `spaCy en_core_web_sm` — dependency parsing and shortest path extraction
- `HuggingFace Transformers` — fine-tuning and inference
- `PyTorch` — model training
- `scikit-learn` — evaluation metrics

---

## How to Run

Open `Drug_Symptom_Treatment_extraction_clean.ipynb` in **Google Colab** with a **T4 GPU** runtime.

Run all cells in order — the notebook handles:
- BC5CDR dataset download automatically
- BioBERT fine-tuning (~15 min on T4)
- Triplet extraction on custom sentences
- Evaluation with Precision, Recall, F1

---
