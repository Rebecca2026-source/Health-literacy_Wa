# Health literacy and smoking among Wa adults — analysis code

Code for the manuscript "Health literacy, smoking status, and nicotine dependence among Wa adults in rural southwest China: a cross-sectional analysis" (PLOS ONE, revised version).

## Files
- `cr2_methods.py` — statistical methods: GLM fitting, CR2 cluster-robust variance with Satterthwaite degrees of freedom, Mancl–DeRouen and CR1 variance, proportional odds and partial proportional odds models, Brant test, random-intercept models, Firth logistic regression, restricted cubic splines.
- `wa_hl_smoking_analysis.py` — all analyses reported in the manuscript and Supporting Information.

## Run
```
pip install numpy pandas scipy matplotlib openpyxl
python wa_hl_smoking_analysis.py <analytic_data.xlsx> <output_folder>
```
Outputs: `results.json`, `results.txt`, `S3_Fig.png`.

## Data
The individual-level data are not public. See the Data Availability Statement of the article for the request procedure. Variable definitions are given in S1 File (data dictionary).
