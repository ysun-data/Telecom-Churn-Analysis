# Customer Churn Prediction: From Model Comparison to Business Action
 
> A telecom company loses a customer every time someone cancels their plan. This project compares six predictive models to find which one to actually trust, uncovers the patterns that separate churners from loyal customers, and answers the question most analyses skip: when exactly should you intervene, and at what cost?
 
---
## What This Project Does
 
| | |
|---|---|
| 📊 **Compared 6 models** | From logistic regression to gradient boosting — does complexity actually pay off? |
| 💡 **Identified churn patterns** | Churners share a distinct set of characteristics across both simple and complex models |
| 💰 **Optimized the intervention threshold** | Because missing a churner and over-alerting both cost money |
| 🔍 **Explained individual predictions** | Using SHAP to open the black box |
 
---
## The Problem
 
Telecom companies spend **$500–800 to acquire a new customer**. Keeping an existing one costs a fraction of that. So when a customer is about to cancel, the smartest move is to catch them early and offer something to stay — a discount, a better plan, a call from support.
 
The challenge is that you can't reach out to everyone. You need a model that tells you *who* is at risk, and a strategy that tells you *when* the math makes sense to act.
 
This project works through that problem in four steps: pick the right model, understand what churners have in common, optimize the intervention threshold, and explain individual predictions. 

---
## The Data
 
**7,032 customers · 19 features · 26.6% churn rate**
 
The dataset comes from IBM's Telco Customer Churn dataset (via Kaggle). Each row is one customer, with features covering:
- **Who they are** — senior citizen, has dependents, has a partner
- **What they use** — internet service type, streaming, tech support, online security
- **How they pay** — contract type, payment method, monthly and total charges
- **How long they've stayed** — tenure (in months)
  
The churn rate of 26.6% reflects real-world distribution — most customers don't leave — so the dataset is moderately imbalanced. This is intentional: fixing it artificially would make the model less useful in practice.
 
Three numeric variables stood out immediately in EDA:
 
<p align="center"><img width="80%" alt="image" src="https://github.com/user-attachments/assets/2be06dce-f887-4b21-a8cc-244f1897db84" /></p>

*Churners tend to have shorter tenure, higher monthly charges, but lower total charges — because they leave before the bill adds up.*
<details>
<summary>📂 Show EDA code</summary>

```r
library(corrplot)
library(e1071)

# Skewness check
skewness(mydata$tenure, na.rm = TRUE)        # 0.24 — roughly symmetric
skewness(mydata$MonthlyCharges, na.rm = TRUE) # -0.22 — roughly symmetric
skewness(mydata$TotalCharges, na.rm = TRUE)   # 0.96 — right-skewed

# Correlation matrix
num_data <- mydata[, c("tenure", "MonthlyCharges", "TotalCharges")]
cor_mat <- cor(num_data, use = "complete.obs")
corrplot(cor_mat, method = "color", addCoef.col = "black",
         tl.col = "black", tl.srt = 45)

# Boxplots
par(mfrow = c(1, 3))
boxplot(tenure ~ Churn, data = mydata,
        main = "Tenure by Churn", xlab = "Churn", ylab = "Months",
        col = c("#7fbfff","#ff7f7f"))
boxplot(MonthlyCharges ~ Churn, data = mydata,
        main = "Monthly Charges by Churn", xlab = "Churn", ylab = "$",
        col = c("#7fbfff","#ff7f7f"))
boxplot(TotalCharges ~ Churn, data = mydata,
        main = "Total Charges by Churn", xlab = "Churn", ylab = "$",
        col = c("#7fbfff","#ff7f7f"))
```

</details>

---
## Step 1 — Which Model Should We Use?

The most common mistake in ML projects is assuming that a more complex model is always better. This project tests that assumption directly.

Six models were trained across three categories — **linear** (logistic regression, LASSO), **nonlinear** (single tree, KNN), and **ensemble** (random forest, boosting) — to cover the full spectrum from simple to complex. 

All were trained on a 70/30 train-test split, with **10-fold cross-validation** used to tune each model's parameters.
 
Three metrics were used to evaluate performance:
- **AUC** — how well the model separates churners from non-churners, regardless of where you set the decision threshold. An AUC of 1.0 is perfect; 0.5 is a coin flip.
- **Sensitivity** — of all customers who actually churned, what fraction did the model catch?
- **Specificity** — of all customers who stayed, what fraction did the model correctly leave alone?

<details>
<summary>📂 Show model training code</summary>

```r
library(caret)

my.ctl <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Logistic regression
log_fit <- train(Churn ~., data = train, method = "glm",
                 family = "binomial", metric = "ROC", trControl = my.ctl)

# LASSO
lassoGrid <- expand.grid(alpha = 1, lambda = 10^seq(0, -2, by = -.1))
lasso_fit <- train(Churn ~., data = train, method = "glmnet",
                   preProcess = c("center","scale"),
                   tuneGrid = lassoGrid, metric = "ROC", trControl = my.ctl)

# KNN
knnGrid <- expand.grid(k = c(5,10,15,20,25,30,35,40,45))
KNN_fit <- train(Churn ~., data = train, method = "knn",
                 preProcess = c("center","scale"),
                 tuneGrid = knnGrid, metric = "ROC", trControl = my.ctl)

# Single tree
treeGrid <- expand.grid(cp = seq(0.001, 0.02, length = 10))
tree_fit <- train(Churn ~., data = train, method = "rpart",
                  tuneGrid = treeGrid, metric = "ROC", trControl = my.ctl)

# Random forest
rfGrid <- expand.grid(mtry = c(2,4,6,8,10),
                      splitrule = "gini", min.node.size = 1)
rf_fit <- train(Churn ~., data = train, method = "ranger",
                tuneGrid = rfGrid, metric = "ROC", trControl = my.ctl,
                importance = "impurity", num.trees = 500)

# Gradient boosting
gbmGrid <- expand.grid(interaction.depth = c(1,2,3),
                       n.trees = c(50,100,150,200,250,300,500),
                       shrinkage = 0.1, n.minobsinnode = 10)
boost_fit <- train(Churn ~., data = train, method = "gbm",
                   tuneGrid = gbmGrid, metric = "ROC",
                   trControl = my.ctl, verbose = FALSE)
```

</details>

---

### Results
<p align="center">
  
| Model | Test AUC | Sensitivity | Specificity |
|---|---|---|---|
| Logistic Regression | 0.8547 | 0.544 | 0.905 |
| LASSO | 0.8513 | 0.491 | 0.918 |
| KNN | 0.8394 | 0.567 | 0.877 |
| Single Tree | 0.8103 | 0.534 | 0.892 |
| Random Forest | 0.8515 | 0.506 | 0.925 |
| **Boosting** | **0.8573** | 0.489 | 0.919 |

</p>

Boosting comes out on top, but only by **0.003 AUC over logistic regression** — a gap that pairwise t-tests across CV folds show is not statistically significant.

<p align="center">
<img width="80%"  alt="image" src="https://github.com/user-attachments/assets/f3f4521d-219a-40cd-87d4-82212f142736" />
</p>

*The confidence intervals overlap heavily. No model clearly dominates.*

However, sensitivity across all models hovers around 0.50 — meaning at the default threshold, roughly half of actual churners are missed entirely. This is exactly the problem that Step 3 addresses.

**The takeaway:**  
When the data structure is relatively simple — driven by a few strong signals — a transparent logistic regression keeps up with the most sophisticated ensemble methods. For a production system where interpretability and maintenance cost matter, that's the practical winner.

<details>
<summary>📂 Show evaluation code</summary>

```r
library(pROC)

# Predict probabilities on test set
log_prob   <- predict(log_fit,   newdata = test, type = "prob")[, "Yes"]
lasso_prob <- predict(lasso_fit, newdata = test, type = "prob")[, "Yes"]
knn_prob   <- predict(KNN_fit,   newdata = test, type = "prob")[, "Yes"]
tree_prob  <- predict(tree_fit,  newdata = test, type = "prob")[, "Yes"]
rf_prob    <- predict(rf_fit,    newdata = test, type = "prob")[, "Yes"]
boost_prob <- predict(boost_fit, newdata = test, type = "prob")[, "Yes"]

# AUC
auc(test$Churn, log_prob)
auc(test$Churn, lasso_prob)
auc(test$Churn, knn_prob)
auc(test$Churn, tree_prob)
auc(test$Churn, rf_prob)
auc(test$Churn, boost_prob)

# Pairwise model comparison
resamp <- resamples(list(logistic = log_fit, LASSO = lasso_fit,
                         KNN = KNN_fit, tree = tree_fit,
                         rf = rf_fit, boosting = boost_fit))
dotplot(resamp, metric = "ROC")
summary(diff(resamp))
```

</details>
---
 
## Step 2 — What Do Churners Have in Common?
 
Before deciding who to target, it helps to understand *why* customers leave. 

Two complementary approaches were used: **relative influence from boosting** (which variables matter most) and **coefficients from logistic regression** (in which direction they matter).
 
<img width="80%"  alt="image" src="https://github.com/user-attachments/assets/b305aec0-4608-4fb2-82f4-19c7d585a816" />

*Month-to-month customers churn at 43% — nearly 15× the rate of two-year contract holders.*
 
### What both models agree on
 
| Predictor | Boosting (relative influence) | Logistic (coefficient) | Plain English |
|---|---|---|---|
| Tenure | 38.95% | -0.059 | Longer-staying customers are much less likely to leave |
| Contract: Two-year | 9.63% | -1.390 | Long-term contracts are the strongest retention tool |
| Contract: One-year | — | -0.687 | Even one-year contracts cut churn risk significantly |
| Internet: Fiber optic | 21.35% | *(not significant)* | Fiber users churn more — possibly a service quality signal |
| Payment: Electronic check | 8.70% | — | Associated with higher churn; may reflect less committed customers |
| Paperless billing | — | +0.344 | Small but independent effect on churn risk |
 
The fact that **tenure alone accounts for ~39% of the model's predictive power** tells a clear story: the first year is the danger zone. If a customer makes it past that window, their churn risk drops sharply.
 
The interesting divergence: fiber optic internet shows up as the second most important variable in boosting (21%), but isn't significant in logistic regression. This suggests its relationship with churn is non-linear — possibly interacting with price or contract type — which boosting can capture but logistic regression cannot. This is likely part of why boosting edges out logistic regression at all.
 
<details>
<summary>📂 Variable importance code</summary>
  
```r
# Boosting variable importance
summary(boost_fit$finalModel)
 
# Logistic regression significant coefficients
summary(log_fit$finalModel)
```
 
</details>
---

## Step 3 — When Should We Intervene?
 
Knowing *who* might churn is only half the problem. The other half is deciding *when* to act — and that depends on the cost of being wrong in each direction.
 
- **Missing a churner (false negative):** the customer leaves, costing ~**$200** in lost lifetime value
- **Flagging a non-churner (false positive):** you send a retention offer to someone who wasn't leaving, costing ~**$20**

Because these costs are asymmetric — a missed churner is 10× more expensive than a false alarm — the default decision threshold of 0.5 is not optimal. Lowering the threshold means flagging more people as at-risk, which catches more real churners at the cost of more unnecessary outreach.
 
The question is: where exactly should we draw the line?
 
<img width="80%"  alt="costcurve" src="https://github.com/user-attachments/assets/84691539-655a-49c3-9d23-95e3c9ede982" />

*Total cost drops sharply as the threshold decreases from 0.5, reaching a minimum at 0.09 before rising again as false alarms pile up.*
 
### Default vs Optimal Threshold
 
| | Threshold | Churners caught | False alarms | Total cost |
|---|---|---|---|---|
| Default | 0.50 | 313 of 577 | 151 | $55,820 |
| **Optimal** | **0.09** | **554 of 577** | **799** | **$20,580** |
 
**Shifting from 0.50 to 0.09 saves $35,240 — a 63% reduction in total cost — without retraining the model.**
 
The tradeoff is real: the optimal threshold flags far more customers (sensitivity jumps from 54% to 96%), which means more retention offers go out. But because each false alarm only costs $20 while each missed churner costs $200, the math strongly favors casting a wider net.
 
This is a critical insight that pure AUC comparisons miss entirely. Two models with identical AUC can have very different business outcomes depending on where you set the threshold.
 
<details>
<summary>📂 Cost analysis code</summary>
  
```r
# Cost assumptions
cost_FN <- 200  # missed churner — lost customer lifetime value
cost_FP <- 20   # false alarm — cost of retention offer
 
# Sweep all thresholds and compute total cost at each
thresholds <- seq(0.01, 0.99, by = 0.01)
 
cost_at_threshold <- function(thresh, probs, labels, fn_cost, fp_cost) {
  predicted <- ifelse(probs >= thresh, 1, 0)
  FN <- sum(labels == 1 & predicted == 0)
  FP <- sum(labels == 0 & predicted == 1)
  return(FN * fn_cost + FP * fp_cost)
}
 
total_costs <- sapply(thresholds, cost_at_threshold,
                      probs = log_prob2, labels = true_labels,
                      fn_cost = cost_FN, fp_cost = cost_FP)
 
# Optimal threshold
optimal_thresh <- thresholds[which.min(total_costs)]  # 0.09
optimal_cost   <- min(total_costs)                    # $20,580
default_cost   <- cost_at_threshold(0.5, log_prob2,
                                    true_labels, cost_FN, cost_FP)  # $55,820
```
 
</details>
---
 
## Step 4 — Why Is This Customer Flagged? (SHAP)
 
Step 2 showed what churners have in common at the population level — patterns across the whole dataset. SHAP goes one level deeper: it explains how each feature affects each *individual* prediction, showing not just what matters on average but how much it mattered for a specific customer.
 
AUC tells you how good a model is overall. But when a model flags a specific customer as high risk, a business needs to know *why* — both to act on it intelligently and to trust the model enough to use it.
 
SHAP (SHapley Additive exPlanations) breaks down each prediction into the contribution of each feature. A positive SHAP value pushes the prediction toward churn; a negative one pulls it away. The color shows the feature value — yellow means high, purple means low.
 
<img width="80%"  alt="shap beeswarm comparison" src="https://github.com/user-attachments/assets/3f5bf89e-81d1-4324-b59f-67620acbddc6" />
*Each dot is one customer. The spread shows how consistently each feature drives predictions across the whole test set.*
 
<img width="80%"  alt="mean SHAP comparison" src="https://github.com/user-attachments/assets/50637240-4178-4273-a68e-642ff5f49986" />
*Bar length = average impact on predictions. Longer bar means the feature moves the needle more.*
 
### What the plots tell us
 
**Tenure** — In both models, longer tenure (yellow) pushes strongly left — lower churn risk. Short tenure (purple) pushes right — higher risk. This is the clearest, most consistent signal in the data.
 
**Internet service** — Fiber optic customers consistently show higher churn risk in both models. This is the second most important feature overall, and its wide spread in the beeswarm suggests the effect varies a lot by customer — likely interacting with price or contract type.
 
**Contract** — Long-term contracts (high value = two-year) pull predictions strongly toward no-churn. Month-to-month customers cluster on the right side of zero.
 
**Monthly charges** — Higher bills (yellow) push toward churn in logistic regression, but the effect is more diffuse in boosting — suggesting the relationship isn't purely linear.
 
### The key finding: 4 out of 5 top features agree
 
| Rank | Logistic regression | Boosting |
|---|---|---|
| 1 | InternetService | tenure |
| 2 | tenure | InternetService |
| 3 | MonthlyCharges | Contract |
| 4 | Contract | PaymentMethod |
| 5 | StreamingTV | MonthlyCharges |
 
**4 of the top 5 features are shared** — InternetService, tenure, MonthlyCharges, and Contract appear in both lists. When a transparent logistic regression and a complex boosting model independently land on the same drivers, that's strong evidence the patterns are real, not artifacts of how one algorithm works. It also means you can trust the simpler model's story when explaining decisions to a non-technical audience.
 
<details>
<summary>📂 SHAP analysis code</summary>
  
```r
library(shapviz)
library(kernelshap)
 
# Background sample for kernelSHAP reference distribution
train_bg  <- train[, !names(train) %in% c("Churn")]
bg_data   <- train_bg[sample(nrow(train_bg), 100), ]
test_features <- test[, !names(test) %in% c("Churn")]
 
# Prediction wrappers
pred_log   <- function(model, newdata) predict(model, newdata, type = "prob")[, "Yes"]
pred_boost <- function(model, newdata) predict(model, newdata, type = "prob")[, "Yes"]
 
# Compute SHAP values (~1-2 min per model)
set.seed(123)
ks_log   <- kernelshap(log_fit2,   X = test_features, bg_X = bg_data,
                        pred_fun = pred_log,   verbose = FALSE)
ks_boost <- kernelshap(boost_fit,  X = test_features, bg_X = bg_data,
                        pred_fun = pred_boost, verbose = FALSE)
 
sv_log   <- shapviz(ks_log)
sv_boost <- shapviz(ks_boost)
 
# Beeswarm plots
library(patchwork)
p_bee <- sv_importance(sv_log,   kind = "beeswarm", max_display = 12) +
           ggtitle("Logistic regression — SHAP beeswarm") +
         sv_importance(sv_boost, kind = "beeswarm", max_display = 12) +
           ggtitle("Boosting — SHAP beeswarm")
print(p_bee)
 
# Bar plots (mean |SHAP|)
p_bar <- sv_importance(sv_log,   kind = "bar", max_display = 12) +
           ggtitle("Logistic — mean |SHAP|") +
         sv_importance(sv_boost, kind = "bar", max_display = 12) +
           ggtitle("Boosting — mean |SHAP|")
print(p_bar)
```
 
</details>
---
 
## Business Takeaways
 
Three things a retention team could act on today, directly from this analysis:
 
- **Focus on the first 12 months.** Tenure is the dominant signal. New customers are disproportionately likely to churn — early engagement programs targeting months 1–6 would address the highest-risk window.
- **Make long-term contracts easier to say yes to.** Month-to-month customers churn at 43% vs 3% for two-year contracts. Even nudging customers toward one-year contracts cuts that risk significantly.
- **Don't wait for the perfect model.** Logistic regression — the simplest model tested — performs within 0.003 AUC of the best. A simple, interpretable model deployed today beats a complex one still being tuned.
---
 
 
