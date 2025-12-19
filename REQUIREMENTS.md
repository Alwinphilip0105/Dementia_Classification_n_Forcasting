# Practical AI Final Project - Requirements
## Individual Research Track | Due: 11:59 PM Friday, 12/19

---

## 📋 DELIVERABLES (What You Submit)

1. **Project Abstract** (1 paragraph)
   - What problem you're analyzing
   - Where to get data
   - What techniques you'll apply
   - What insights you plan to extract
   - *Due with final submission (discuss early with instructor if unsure)*

2. **Project Notebook** (Jupyter)
   - Main artifact of the project
   - Professional report on problem + insights gained
   - Techniques + visualizations
   - *Must be saved in run state (results visible without re-running)*

3. **Project Modules** (Python packages)
   - Complex functions in separate modules
   - Import into notebook
   - Minimal code in notebook itself

4. **README.md**
   - Purpose
   - Usage Instructions (clone, install, use with code snippets)
   - Known Issues
   - Feature Roadmap
   - Contributing
   - License
   - Contact

5. **DATA_CARD.md**
   - Where data came from
   - Purpose of dataset
   - Author
   - Owner
   - License
   - Significant processing performed

6. **MODEL_CARD.md**
   - Final metrics (domain-specific)
   - How model was trained/evaluated
   - Known limitations

---

## 🎯 GRADING BREAKDOWN (100 points total)

### Originality (25%)
- **Problem Choice** (5 pts): Why is this problem interesting to your field?
- **Original Steps** (5 pts): Combine 2+ tools in unique ways (not just default sklearn models)
- **Novel Insights** (5 pts): Extract at least 1 novel insight (Type 2, 3, or 4)
- **Actionable Insights** (5 pts): 3+ actionable insights (reader can take action)
- **Original Exposition** (5 pts): Explain insights in your own words, original manner

### Software Hygiene (25%)
- **Modules** (5 pts): Functions >20 lines in modules, not in notebook
- **Portable Code** (5 pts): No local hardcoded paths, runs in any environment
- **Documentation** (5 pts): All functions have docstrings, professional tone
- **Styling** (2.5 pts): Formatted with black
- **Commit Messages** (2.5 pts): High quality messages (reads "When applied, this commit will...")
- **Variable Names** (2.5 pts): Descriptive nouns, include units for quantities, no single letters
- **No Dead Code** (2.5 pts): No commented-out code

### ML Best Practices (35%)
See detailed section below

### Basics (15%)
- **Functions/Lambdas** (3 pts): Correct implementation, no errors
- **Conditionals/Loops** (3 pts): Correct implementation
- **Data Structures** (3 pts): Correct use of lists, dicts, tuples, etc.
- **NumPy** (3 pts): Use array operations, not loops
- **DataFrames** (3 pts): Proper use of apply(), to_list(), masks, etc.

---

## 🔴 ML BEST PRACTICES (35%) - DETAILED

### Basic Basics (Mandatory)
- [ ] Do NOT train on test data
- [ ] Account for data drift (describe how you'll detect/remove drifted samples or retrain)
- [ ] Problem fits your model's learning capacity with available resources
- [ ] Pre and post guardrails for input/output validation
- [ ] Use Occam's razor (narrow task focus)

### Basic Basic Tuning (Mandatory)
- [ ] Visualize data to confirm labels are accurate, edit bad labels
- [ ] Remove bad/suspect data
- [ ] Define loss function using domain knowledge ("definition of good")
- [ ] Define learning paradigm early (supervised/unsupervised/RL/hybrid)
- [ ] Tune learning rate first in hyperparameter tuning

### Project Planning (Mandatory - 10 pts)
- [ ] Choose planning paradigm (V-model, Russell & Norvig, Goodfellow, Google) and describe phases
- [ ] Discuss if ML is best solution vs non-ML baselines
- [ ] Define definition of good performance + performance goals BEFORE starting
- [ ] Specify which problem you're solving for users, which parts use ML
- [ ] Assess ethics (autonomous weapons? privacy? security? fairness? trust? safety? harm?)
- [ ] Coverage-accuracy tradeoffs?
- [ ] Data collection feasibility? How much data needed?
- [ ] Expected queries per second?
- [ ] Budget and staff needed to productionize?
- [ ] Does project show promise for early productionization? What results before production?
- [ ] Plan for user/stakeholder feedback?

**Include 1-2 paragraph discussion in your notebook**

### Architecture (Mandatory)
- [ ] Separate training and inference code
- [ ] Serialize configuration with YAML or JSON
- [ ] Serialize models with ONNX
- [ ] Define models as classes extending PyTorch's nn.Module
- [ ] If custom dataset, extend PyTorch's Dataset class

### Data Pipeline (Mandatory)
- [ ] Remove exceptional cases
- [ ] Feature engineering where appropriate
- [ ] Use unsupervised learning as first step if data unlabeled/suspect labels
- [ ] Annotate ambiguous labels for special analysis
- [ ] Note class imbalance, under/oversample as needed
- [ ] Exploratory data analysis (correlations, visualizations, PCA, T-SNE, autoencoders)
- [ ] Consider discretizing features to remove noise
- [ ] Consider ICA for noise removal

### Training Models (Mandatory)
- [ ] Evaluate multiple architectures
- [ ] Attempt multiple experiments, document all parameters
- [ ] Use seeds for reproducibility
- [ ] Use data loaders (mandatory for PyTorch)
- [ ] Change single hyperparameters in small ways, record progress
- [ ] Discuss no free lunch theorem
- [ ] Consider hybrid architectures (deep + statistical)
- [ ] Tune hyperparameters to increase/decrease model capacity

### Explainability (Mandatory)
- [ ] Consider performance vs explainability tradeoffs, discuss in report
- [ ] Discuss type of explainability suited for problem (strict interpretability needed?)
- [ ] Consider multiplicity of good models problem (if many solutions, which most robust/explainable/reliable?)
- [ ] Consider surrogate models where appropriate

### Robustness and Hardening (Techniques - apply as appropriate)
- [ ] Augment data for robustness to transformations
- [ ] Select architectures deliberately with domain knowledge/comparative testing
- [ ] Engineer robust features, eliminate noisy ones
- [ ] Employ filters (averaging, advanced techniques)
- [ ] Remove noise below thresholds with nonlinearity
- [ ] Detect and remove drifted/out-of-distribution samples
- [ ] Regularize with custom loss if needed
- [ ] Sanitize data (remove low quality, anomalous, malicious inputs)
- [ ] Consider adversarial training or model distillation

### Test and Evaluation (Mandatory)
- [ ] Run explainability on candidate models
- [ ] Run conformance test after final models cut
- [ ] Robustness test against challenging transforms (blur, noise, darkness, etc.) if appropriate
- [ ] Extract basic metrics: loss, confusion matrix, false positives/misses
- [ ] Compute final metrics in problem domain (what stakeholder cares about, not just ML engineer metrics)
- [ ] Test and compare against baseline

### Discussion Section Requirements
- [ ] Discuss at least one item from EACH applicable section (Project Planning, Architecture, Data Pipeline, Training, Explainability, Robustness, Test & Eval)
- [ ] Just a few sentences per item

---

## 📊 TOOLS (5+ Required)

Used in class:
- PyTorch (deep learning)
- Captum (explainability)
- Scikit-Learn (ML)
- SciPy (optimization)
- Pandas (data management)
- ONNX (model saving)
- Seaborn/Matplotlib (visualization)
- Adversarial Robustness Toolkit / Cleverhans (robustness testing)

**You need at least 5 distinct tools**

---

## 🧪 TECHNIQUES (5+ Required)

From class content:
- Self-supervised learning
- Transfer learning
- Temporal modeling
- Multi-task learning
- Explainability/interpretability
- Baseline comparisons
- Contrastive learning
- Data augmentation
- Ensemble methods
- Regularization techniques
- Feature engineering
- Dimensionality reduction (PCA, t-SNE)
- Anomaly detection
- Class balancing

**You need at least 5 distinct techniques**

---

## 📝 NOTEBOOK STRUCTURE (Recommended Sections)

- [ ] **Title**: Descriptive
- [ ] **Abstract**: Problem, methods, results (1 para)
- [ ] **Introduction**: What are you trying to achieve? What methods?
- [ ] **Problem Addressed**: What causes pain? Who suffers?
- [ ] **Motivation**: Why important? What's the payoff?
- [ ] **Previous Work**: Literature review + citations
- [ ] **Project Schedule and Budget**: Planning discussion (1-2 para mandatory)
- [ ] **Technical Approach**: Why justified?
- [ ] **Main Results**: Present results
- [ ] **Future Work**: Next steps, roadmap, research needed
- [ ] **Discussion**: Summarize findings, implications
- [ ] **ML Best Practices Discussion**: Hit ≥1 item from each applicable section

**Requirements:**
- Full sentences and paragraphs (3-10 sentences per paragraph)
- Markdown headings for sections
- LaTeX for equations
- Guide readers through material
- Professional tone (not exploratory)

---

## 🎨 VISUALIZATION ANTI-PATTERNS TO AVOID

❌ **Don't:**
- Plot without title
- Plot without axis labels
- Overlapping ticks/text
- Labels too small
- Too many x-ticks
- Multiple relationships without legend
- Subplots that run into each other
- Dates in non-date format
- Unnecessary gridlines
- Noisy data without trend extraction
- Axes in unnatural units
- Statistical data without uncertainty
- 1D scatter plots (use histograms/violin plots instead)
- Too many subplots without expanding figure
- Correlation matrices without investigation

❌ **Math Notation:**
- Don't write `e^(sin(x))`, use proper LaTeX: $e^{\sin(x)}$

---

## 💾 REPOSITORY STRUCTURE

```
.devcontainer/
projectname/           # Your Python package
  subpackage/
    __init__.py
    module.py
  __init__.py
  module.py
notebooks/
  final_project.ipynb  # Main deliverable
tests/
  test_*.py
.gitignore
LICENSE
README.md
DATA_CARD.md
MODEL_CARD.md
poetry.lock
pyproject.toml
```

---

## 🔄 NOVEL INSIGHT TYPES

- **Type 1** (Not Novel): Old problem + old method (avoid)
- **Type 2** (Novel): Old problem + new method
- **Type 3** (Novel): New problem + old method
- **Type 4** (Novel): New problem + new method

**You need at least one Type 2, 3, or 4 insight**

### What Makes an Insight Actionable?
Ask yourself: *Can the reader take action from this? What action?*
- Must be specific
- Must be implementable
- Must be grounded in your analysis

### What Makes an Insight Original?
Ask yourself: *Would someone hire me over an AI tool or another person to produce this?*
- Only you could have produced it
- Breaks down unsolved problems into subparts
- Synthesizes your unique experience

---

## ⚠️ ANTI-PATTERNS TO AVOID (2.5 pts each)

### AI/ML Anti-Patterns
- [ ] Don't start without defining the problem
- [ ] Don't start ML project without assessing if ML is best solution
- [ ] Don't start without baseline to improve against
- [ ] Don't train complicated model without evaluating simple model first
- [ ] Don't assume waterfall will work for MLOps
- [ ] Don't assume human labels are infallible/unbiased
- [ ] Don't use ML without drift detection + retraining mechanism
- [ ] Don't couple training and inference code
- [ ] Don't use insecure formats for model serialization
- [ ] Don't use unclean data
- [ ] Don't use data without checking for ambiguous/inseparable cases
- [ ] Don't ignore no free lunch theorem—narrow task as much as possible
- [ ] Don't tune multiple hyperparameters at once
- [ ] Don't cut models without running explainability
- [ ] Don't wait until final training to check explainability
- [ ] Don't show performance curves without repeating to ensure reproducibility
- [ ] Don't push to production without conformance test
- [ ] Don't show results to stakeholders without domain-specific system-level metrics

### Technical Writing Anti-Patterns
- [ ] No spelling mistakes
- [ ] No incomplete sentences
- [ ] Don't use exploratory tone ("we tried X, didn't work, so we tried Y")
- [ ] Make goal of each section clear

### Software Anti-Patterns
- [ ] Don't develop large code blocks in notebook
- [ ] Don't use local machine-specific paths
- [ ] Don't use for loops where NumPy works
- [ ] Don't write lines >120 characters (>80 with black)
- [ ] Don't comment every line
- [ ] Don't use single letter variable names
- [ ] Don't leave commented-out code

### General Anti-Patterns
- [ ] Don't leave sections incomplete
- [ ] Don't report success with poor hyperparameter search or increasing error curves
- [ ] Don't report success with clearly overfit model
- [ ] Don't leave dead code or broken cells
- [ ] Don't show flattening error curve without increasing model capacity
- [ ] Don't choose problem outside your ability to solve
- [ ] Don't forget data visualization to check labels/loading
- [ ] Don't skip robustness testing or explainability
- [ ] Don't forget to document hyperparameters tuned
- [ ] Don't forget to compare improvements to baseline

---

## 🤖 POLICIES

### Collaboration
- ✅ Can collaborate to debug
- ❌ DO NOT use same code as another student
- ❌ DO NOT extract same insights as another student
- ❌ DO NOT collaborate outside class
- ❌ DO NOT submit work that isn't yours
- **Violation = Zero for final project**

### AI Tool Use
- ✅ Can use AI to help write code
- ✅ Can use AI as a tool to expand your insights
- ❌ DO NOT submit AI code without adding original work
- ❌ DO NOT use AI code without verifying it works
- ❌ DO NOT violate "Veterinary Dentists Law": have domain knowledge + tool knowledge
- ❌ DO NOT have same code as another student (even with trivial modifications)
  - 1st occurrence: -5 pts
  - 2nd occurrence: -10 pts
  - 3rd occurrence: -20 pts, etc.

---

## ✨ HINTS FOR SUCCESS

1. **Start with the problem, not the data.** Define what pain you're solving first.

2. **Pick a problem that matters to you.** Genuine interest shows in the work.

3. **Literature review is crucial.** Show you understand what others have done.

4. **Simple baseline first.** Always compare against something simple before complex models.

5. **Reproducibility is everything.** Document hyperparameters, use seeds, save configs.

6. **Explainability isn't optional.** Run it even if not required for your problem—it guides model selection.

7. **Your insights are the grade.** Deliverables are the vehicle; insights are the cargo.

8. **Professional tone throughout.** This is a final deliverable, not a lab notebook.

9. **Think like an employer.** Would you hire someone who produced these insights?

10. **Schedule office hours early.** Especially if unsure about problem choice.

---

## 📅 TIMELINE

- **Now**: Choose problem, discuss with instructor if unsure
- **By 12/19 11:59 PM**: All deliverables submitted (abstract + notebook + modules + docs)
- **Intermediate**: Develop, test, verify against checklist regularly
- **Final review**: Check this summary checklist before submitting

---

## 🚀 FINAL SUBMISSION CHECKLIST

- [ ] All 6 deliverables present (abstract, notebook, modules, README, DATA_CARD, MODEL_CARD)
- [ ] Notebook runs without errors in saved state
- [ ] No hardcoded paths
- [ ] ≥5 tools used
- [ ] ≥5 techniques used
- [ ] ≥1 novel insight extracted
- [ ] ≥3 actionable insights presented
- [ ] No dead code or commented sections
- [ ] Code formatted with black
- [ ] All functions have docstrings
- [ ] Commit messages are high quality
- [ ] README complete
- [ ] DATA_CARD and MODEL_CARD complete
- [ ] Professional writing throughout
- [ ] Visualizations follow best practices
- [ ] Git submitted (no points deducted for mechanics)

---