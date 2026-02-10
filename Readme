Early-Stage Fairness Analysis in Machine Learning for Requirements Engineering
Overview

Machine learning (ML) has become an integral part of software systems, but biases in these models, especially during requirements engineering, pose significant challenges. Existing fairness frameworks, such as ReFair, rely on shallow machine learning models for classifying sensitive features in user stories. While these frameworks demonstrate some success, they struggle to capture complex linguistic patterns and contextual nuances.

This project introduces a deep learning-based framework to improve early-stage fairness analytics, enhancing the identification of sensitive features from user stories. By leveraging advanced word embeddings and sequential deep learning models, the framework aims to reduce bias and improve the accuracy and contextual understanding of ML systems from the earliest stages of development.

Problem Statement

Bias in ML models introduced during requirements engineering can propagate throughout the software lifecycle, leading to unfair system behavior. Key limitations of existing approaches include:

Limited research on early-stage fairness: Most work focuses on model training or post-deployment monitoring rather than identifying biases during requirements specification.

Reliance on shallow models: Current frameworks use simple models that cannot fully capture the complexity of natural language in user stories.

Avoidance of advanced AI techniques: Deep learning methods are often skipped due to concerns about computational cost or interpretability, limiting the detection of nuanced biases.

Objectives

The project aims to:

Develop a robust framework for detecting sensitive features in user stories.

Utilize advanced word embeddings (ALBERT, RoBERTa, DistilBERT) for contextual understanding.

Implement deep learning architectures (LSTM, BiLSTM, GRU, BiGRU) to capture sequential dependencies.

Improve classification accuracy and bias detection compared to shallow models.

Enhance fairness in ML pipelines by addressing biases early in the development lifecycle.

Methodology
1. Data Preprocessing

User stories are collected and cleaned for text analysis.

Tokenization and normalization are applied to prepare the text for embedding generation.

2. Feature Representation

Word Embeddings: Advanced embeddings like ALBERT, RoBERTa, and DistilBERT are used to create rich contextual representations of user stories.

These embeddings capture subtle semantic and linguistic nuances for improved classification.

3. Deep Learning Models

LSTM & BiLSTM: Retain and process long-range dependencies; BiLSTM processes data in both forward and backward directions.

GRU & BiGRU: Efficient alternatives to LSTMs while maintaining sequential learning capabilities. BiGRU enhances contextual understanding by considering past and future sequences.

4. Classification and Sensitive Feature Detection

The model classifies user stories based on task type and application domain.

Sensitive features are extracted from the intersection of domain-specific and task-specific features.

The framework generates a recommended set of sensitive features to minimize biases in software requirements.

Key Features

Advanced NLP techniques for contextual embedding.

Sequential deep learning models for improved feature detection.

Robust against noisy or complex textual data.

Reduces biases at the earliest stages of ML system development.

Provides actionable insights for fairer requirements engineering.

Tools & Technologies

Programming Language: Python

Deep Learning Frameworks: TensorFlow / PyTorch

Word Embeddings: ALBERT, RoBERTa, DistilBERT

Sequential Models: LSTM, BiLSTM, GRU, BiGRU

Evaluation Metrics: Classification accuracy, bias detection metrics
