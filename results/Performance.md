# Model Performance Analysis

## Overall Performance

The fine-tuned **ResNet-50 model achieved strong overall performance** on the evaluation dataset, reaching an accuracy of **95.06%** and correctly classifying **1,521 out of 1,600 images**.

| Metric              |      Score |
| ------------------- | ---------: |
| **Test Loss**       |     0.2484 |
| **Accuracy**        | **95.06%** |
| **Macro Precision** |     0.9543 |
| **Macro Recall**    |     0.9506 |
| **Macro F1-Score**  |     0.9499 |
| **Macro AUC-ROC**   | **0.9897** |

The close agreement between macro precision, recall, and F1-score indicates that the model performs relatively consistently across the four classes. Additionally, the **AUC-ROC score of 0.9897** demonstrates excellent overall class separability based on the model's predicted probabilities.

---

## Per-Class Performance

| Class       |  Precision |     Recall |   F1-Score | Samples |
| ----------- | ---------: | ---------: | ---------: | ------: |
| **Class 0** | **0.9970** |     0.8350 |     0.9088 |     400 |
| **Class 1** |     0.8921 | **0.9925** |     0.9396 |     400 |
| **Class 2** |     0.9356 |     0.9800 |     0.9573 |     400 |
| **Class 3** | **0.9925** | **0.9950** | **0.9938** |     400 |

### Class 0

Class 0 represents the model's main area of weakness. Although its **precision is extremely high at 99.70%**, its **recall is only 83.50%**.

This means that when the model predicts Class 0, it is almost always correct. However, the model fails to identify a noticeable number of actual Class 0 samples.

The confusion matrix shows that:

* **39 Class 0 images** were predicted as Class 1.
* **27 Class 0 images** were predicted as Class 2.

Therefore, **66 out of 400 Class 0 samples** were misclassified. The main challenge for the model is distinguishing Class 0 from Classes 1 and 2.

### Class 1

Class 1 achieves an excellent **recall of 99.25%**, meaning that nearly all actual Class 1 images are correctly identified.

However, its precision is lower at **89.21%**, primarily because a significant number of Class 0 images are incorrectly predicted as Class 1.

This suggests that the model is highly sensitive to Class 1 but occasionally assigns the Class 1 label to samples belonging to other classes.

### Class 2

Class 2 demonstrates strong and balanced performance, with:

* **93.56% precision**
* **98.00% recall**
* **95.73% F1-score**

Only a small number of actual Class 2 samples are misclassified, with most predictions correctly assigned to their true class.

### Class 3

Class 3 is the best-performing category, achieving an F1-score of **99.38%**.

Out of 400 samples:

* **398 were correctly classified**
* Only **2 samples were predicted as Class 1**

This indicates that Class 3 has a highly distinguishable visual representation within this dataset.

---

## Confusion Matrix Analysis

The confusion matrix reveals a clear pattern in the model's errors:

```text
                Predicted
              C0    C1    C2    C3
Actual C0    334    39    27     0
Actual C1      0   397     0     3
Actual C2      1     7   392     0
Actual C3      0     2     0   398
```

The majority of predictions lie along the main diagonal, confirming the model's strong classification performance.

The most significant source of error is:

> **Class 0 being confused with Class 1 and Class 2.**

In contrast, confusion involving Class 3 is extremely limited.

This pattern suggests that the model has learned strong representations for Classes 1, 2, and especially 3, but the visual characteristics of Class 0 overlap more substantially with other categories in the learned feature space.

---

## Key Findings

* The model achieves a strong overall accuracy of **95.06%**.
* **Class 3 is classified almost perfectly**, with an F1-score of **99.38%**.
* **Classes 1 and 2 also demonstrate strong performance**, particularly in recall.
* The primary weakness is **Class 0 recall**, which falls to **83.50%**.
* Most classification errors originate from actual Class 0 samples being predicted as **Class 1 or Class 2**.
* The **AUC-ROC of 0.9897** indicates excellent overall ability to separate the four classes.

## Conclusion

Overall, the fine-tuned ResNet-50 model demonstrates **strong performance on the current evaluation dataset**, with highly consistent results across most classes. The main opportunity for improvement is increasing the recall of Class 0 and reducing its confusion with Classes 1 and 2.

Possible future improvements include further investigating the Class 0 samples, examining dataset quality and class-specific visual overlap, tuning the augmentation strategy, adjusting the fine-tuning procedure, or using techniques designed to focus training on difficult or frequently confused examples.

While these results are promising, strong performance on this test dataset alone is **not sufficient to establish clinical reliability**. Further validation on independent and representative datasets would be necessary before making real-world medical claims.
