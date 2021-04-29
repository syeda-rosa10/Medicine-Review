# Medicine-Review
The project was to predict if a drug has side-effects or not based on the user review.

Based on the review, the medicine was scored 1-10, 1 being that the medicine had most side-effects, 10 being least side-effects.
then the scores were converted to binary, scores 1-7 were labelled as '0' and above 7 labelled as '1'.

Then TF-IDF vectorisation was performed to convert review.
Smote was applied to balance the data.
Models used- Logistic, SVC, Random Forest classifier.

Random Forest classifier was choosen as it gave best accuracy.




