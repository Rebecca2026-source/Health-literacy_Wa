# Health literacy and smoking among Wa adults — analysis code

Code for the manuscript "Health literacy, smoking status, and nicotine dependence among Wa adults in rural southwest China: a cross-sectional analysis" (PLOS ONE, revised version).

## Files
- `cr2_methods.py` — statistical methods: GLM fitting, CR2 cluster-robust variance with Satterthwaite degrees of freedom, Mancl–DeRouen and CR1 variance, proportional odds and partial proportional odds models, Brant test, random-intercept models, Firth logistic regression, restricted cubic splines.
- `wa_hl_smoking_analysis.py` — all analyses reported in the manuscript and Supporting Information, including the full regression results in S6 Table.

## Model names
In the code, `CORE` is M3 (main model) and `EXT` is M4 (extended model). M3 adjusts for age, sex, education, household income, occupation and marital status. M4 also adjusts for regular drinking, BMI, PHQ-9, GAD-7 and chronic disease.

## Run
```
pip install numpy pandas scipy matplotlib openpyxl
python wa_hl_smoking_analysis.py <analytic_data.xlsx> <output_folder>
```
Outputs: `results.json`, `results.txt`, `S3_Fig.png`. The key `S6` in `results.json` holds the full results for all covariates in M3 and M4 (S6 Table).

## Data
The individual-level data are not public. See the Data Availability Statement of the article for the request procedure. Variable definitions are given in S1 File (data dictionary).
