## **MusQuE Training**

MusQuE model is designed for event-specific document ranking. You can use the provided data samples in this repository to quickly get started with training the model.
Alternatively, by accessing the main Ms-Marco dataset, you can create the full dataset following the main settings of Ms-Marco.
To do so, you need the query identifiers (qids) from event-queries.tsv file provided in the Ms-Marco-Event dataset.

## **MusQuE Evaluation**
For evaluating the performance of MusQuE use the stored trained model. The output of MusQuE_evaluation.py script is a file including the scores for each query and document ID, along with the corresponding ground truth label (0 or 1). This output can be used later for computing different evaluation metrics.
