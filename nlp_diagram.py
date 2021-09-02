from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS
from diagrams.aws.network import Route53

with Diagram("Natural Language Processing Overview", show=False, direction='TB') as d:
    data = ECS('Data')

    with Cluster('Preprocessing'):
        remove_punct = ECS('Remove punctuation')
        tolken =  ECS('Tolkenization')
        stopwords = ECS('Remove Stopwords')
        remove_punct  >> tolken >> stopwords

        stem = ECS('Stemming')
        lemma = ECS('Lemmatizing')

        stopwords >> stem
        stopwords >> lemma

    clean_data = ECS('Clean Data')
    stem >> clean_data
    lemma >> clean_data

    with Cluster('Feature Engineering'):
        with Cluster('Vectorize'):
            vectorize = [ECS('Count Vectorization'),ECS('N-Grams'),ECS('TF-IDF')]

        transformation = ECS('Box-Cox Power Transformation')
        standardizing_data = ECS('Standardizing data')
        feat_creation = ECS('Feature Creation')

    clean_data >> vectorize
    clean_data >> transformation
    clean_data >> standardizing_data
    clean_data >> feat_creation

    select_model = ECS('Select Model')
    vectorize >> select_model
    clean_data >> select_model
    clean_data >> select_model
    clean_data >> select_model

    with Cluster('Models'):
        ml = [ECS('Random Forest'),
            ECS('Gradient Boosting'),
            ECS('RNNs')]

    with Cluster('Metrics'):
        model_evaluation = [ECS('Accuracy'),ECS('Recall'),ECS('Precision')]

    with Cluster('Machine Learning'):
        ml_model = ECS('ML Model')
        cross_validation = ECS('Cross Validation')
        metrics = ECS('Metrics')

        select_model >> ml_model
        ml_model >> cross_validation
        cross_validation >> metrics
        metrics >> ml_model

    ml_model >> ml
    metrics >> model_evaluation

    final_model = ECS('Final Model')
    ml_model >> final_model

    with Cluster('Applications'):
        applications = [ECS('Sentiment Analysis'),
                        ECS('Topic Modeling'),
                        ECS('Text Classification'),
                        ECS('Sentence Segmentation')]
    final_model >> applications

    data >> remove_punct

d
