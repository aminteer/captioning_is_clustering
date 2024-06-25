# Captioning is Clustering: Automated Image Classification based on Image Captioning Text for Accessibility

## ABSTRACT
This report proposes a machine learning project that utilizes the Flickr8k dataset, consisting of images and five variations of corresponding captions, to develop a model capable of automatically classifying new images into predefined categories based on generated captions. The first step in this effort will be to identify clusters based on the caption text.  The primary aim is to enhance accessibility by providing automated image categorization. This can be beneficial for visually impaired users, for example. By leveraging techniques in image captioning and natural language processing, this project seeks to create an automated and adaptable system for image classification. This report outlines the existing research in the field, details the planned steps for the project, including exploratory data analysis (EDA), model building, training, and evaluation, and provides a proposed timeline for the project.  
## 1.	INTRODUCTION
The growing need for accessibility solutions has prompted significant advancements in machine learning and artificial intelligence. Among these, the automatic classification of images based on generated captions holds particular promise. The Flickr8k dataset, which contains 8,000 images each paired with five different captions, provides a starting foundation for this project. This project aims to develop a model that can use generated captions to classify new images into categories established by the clustering and categories process, thereby assisting users, such as ones with visual impairments, by providing categorized information about images.
## 2.	RELEVANT LITERATURE REVIEW
Recent years have seen considerable progress in the fields of image captioning and image classification. Research has demonstrated that convolutional neural networks (CNNs) and recurrent neural networks (RNNs) can be effectively combined to generate descriptive captions for images (Vinyals et al., 2015). Additionally, transformer models, such as the Vision Transformer (Dosovitskiy et al., 2020), have shown promise in handling image data for various tasks, including caption generation and classification.
Studies have also explored the use of the Flickr8k dataset for training models in image captioning. For instance, Karpathy and Fei-Fei (2015) used deep neural networks to align image regions with natural language phrases, achieving significant results in caption generation. Moreover, recent advances in accessibility research have highlighted the potential benefits of automated image description systems for visually impaired users (Gurari et al., 2020).

## 3.	PROJECT OVERVIEW
### 3.1	EXPLORATORY DATA ANALYSIS 
The initial phase of the project involves a thorough exploratory data analysis (EDA) of the Flickr8k dataset. This includes:
1. Data Inspection: Understanding the structure and content of the dataset, including image dimensions, caption formats, and any missing values.
2. Visualization: Creating visualizations to explore the distribution of image features and caption lengths.
3. Caption Analysis: Analyzing the captions to identify common themes, keywords, and patterns that can inform the categorization process.
Initial analysis of the dataset shows that image size is 299 x 299 with 3 channels (RGB) for the 8,000 images.  Each image has a corresponding five captions for training and validation purposes.
Caption length by characters was analyzed to determine if there adjustments needed for too low informational value examples.   The analysis showed a distribution with a longer tail of wordy captions.
 
Caption length distribution.
Analysis indicates no missing values.  Some captions were either very short to be of informative value or far too long for useful analysis.  After tokenization, captions that have less than five tokens or more than 25 tokens were removed.
![image](https://github.com/aminteer/captioning_is_clustering/assets/30326111/270ad819-afd2-4af5-96dd-5e6c9ae1d794)


### 3.2	Model Building and Training Plan
 The model building phase involved several steps:
1. Preprocessing: Preparing the images and captions for model training, including resizing images, tokenizing captions, and creating training and validation splits.
2. Feature Extraction: Using a pre-trained CNN (such as InceptionV3 or ResNet) to extract features from the images.
3. Caption Generation: Implementing an encoder-decoder architecture with an attention mechanism to generate captions from image features. The encoder will be based on a CNN, and the decoder will be an RNN or a transformer model.
4. Clustering Model: Cluster data using Expectation Maximization (EM) methods as a starting point.  Using EM methods provides the probability of belong to each of the clusters for each caption processed.  This could be useful for providing users options ranked by the probability order.  If EM methods do not produce useful results, then we will attempt K-Means clustering for comparison 
As part of the model building processing, image preprocessing and augmentation was performed to improve the ability of the resulting model to generalize when new images are inferenced to generate captions.  Flip, rotation, and contrast random adjustments were completed.  Examples are shown in the image.
![image](https://github.com/aminteer/captioning_is_clustering/assets/30326111/e35ca48d-8fc7-4aaa-b151-fbfdd5097702)


 
Image augmentation example
 
Actual image from training set, post preprocessing

Additional preprocessing for the captions included text formatting – removing unwanted characters, lower case, etc
Example: Strip Characters "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~”

Image feature extraction was performed with a type of Convolutional Neural Network (CNN), called Efficient Net.  Specifically, we decided to use EfficientNetB0.  EfficientNetB0 is a baseline model in the EfficientNet family.  It typically exhibits excellent performance and efficiency in terms of parameter size and computational cost. EfficientNet models were developed by Google and use a scaling method that uniformly scales all dimensions of depth, width, and resolution using a compound coefficient.
 
EfficientNetB0 model architecture
Captioning text was transformed into a vector for initial cluster analysis using text vectorization techniques.  Gaussian Mixture Models were then used to identify clusters.  In order to visualize the results, dimensionality reduction was performed with UMAP and plotted with clusters identified by colors.  As can be seen in the plot, the resulting clusters and visualization does not align well.  Additional work was needed. We later decided to use embedding models as opposed to the simple text vectorization performed initially.
 
Training set captions clustered using Gaussian Mixture Model and dimensionality reduction with UMAP.  Each color represents a cluster identified with GMM.

Example of captions clustered in Cluster 1 with the initial text vectorization and EM clustering method:
Cluster 1
<start> A large bird swooping down towards the ground . <end>
<start> An orange canoe aimed toward two other canoes on a lake . <end>
<start> A man wearing US flag boxer shorts is standing on a stairway . <end>
<start> a single person squated down at the edge of a pier overlooking a lake <end>
<start> A girl is posed over a statue of a dangerous animal . <end>
As may be noticed, the examples are not clearly coherent as belonging to a common group.  Based on the clustering visualization and the review of clustered captions, it is clear more improvement was needed.
For the first step to improve these results, we decided to switch to K-Means for clustering methodology.  Using UMAP for dimensionality reduction, results are visualized below.
 
Text vectorization with K-Means clustering.  Dimensionality reduction with UMAP.	
Example of captions from Cluster 0 with K-Means
Cluster 0
<start> Lambs on a grassy hill . <end>
<start> Man in a blue coat holding a sign . <end>
<start> A young man holds up a hand written sign next to a woman with a multicolored umbrella . <end>
<start> The three girls sat on the beach . <end>
<start> A girl with a black purse sitting on a wooden bench . <end>
In order to improve clustering results, we decided to approach this with an embeddings process instead of a simple text vectorization.  After reviewing options, we determined using a transformer based model from the sentence transformers library, paraphrase-MiniLM-L6-v2.  The 'paraphrase-MiniLM-L6-v2' model from the Sentence Transformers library in Python is a state-of-the-art transformer-based model designed for efficient and effective natural language processing tasks, particularly those involving sentence embeddings. Developed as part of the MiniLM (Mini Language Model) series, this model leverages a lightweight architecture that balances performance and computational efficiency.
	The 'paraphrase-MiniLM-L6-v2' model is built on top of the transformer architecture, which has shown huge improvement in NLP tasks through its self-attention mechanism. The model utilizes a transformer architecture with six layers (L6), which processes input sentences in parallel, enabling efficient handling of large datasets.  Each layer employs self-attention mechanisms to capture relationships between words in a sentence, regardless of their positions.
	The primary use of 'paraphrase-MiniLM-L6-v2' is to generate dense vector representations (embeddings) of sentences. These embeddings capture the semantic meaning of sentences in a continuous vector space.  This is ideal for what we are developing in this project.
	Image captioning evaluation often requires measuring the semantic similarity between generated captions and reference captions. The 'paraphrase-MiniLM-L6-v2' model does well at generating embeddings that capture semantic nuances, making it ideal for this task. The model's training on paraphrased datasets ensures that it can effectively handle different phrasings of the same underlying concept. This robustness is particularly beneficial for image captioning, where multiple valid descriptions can exist for a single image.
	We also decided to cluster using K-Means methodology in order to ensure every new image and caption would be assigned to one of the five clusters.  The combination of using a transformer model to generate embeddings and K-Means for clustering showed immediate improvements.
	We decided to perform dimensionality reduction using tSNE method. Results are shown in the image below.
 
Transformers Embeddings with K-Means clustering.  Dimensionality reduction with t-SNE.
Example of captions from Cluster 0 when transformer embeddings and K-Means clustering was used:
<start> A woman in a short skirt holds a plastic bottle as she walks down a street with auto and pedestrian traffic 
<start> A group of women riding horses while holding the flags of multiple nations <end>
<start> A laughing woman holding a little girl . <end>
<start> A woman and a little girl sit on desert rock . <end>
<start> A girl wearing red and blue clothing poses for a man kneeling to take her picture . <end>
On review, the visualization and caption review both show significant improvement in clustering coherence.
### 3.3	Evaluation Plan
The evaluation phase will focus on thoroughly assessing the performance and effectiveness of both the caption generation and image classification models. This phase is critical to ensure that the resulting system meets the defined success criteria and performs reliably in real-world applications. The evaluation will be conducted through the following steps:
1. Caption Quality:
   The quality of the generated captions will be evaluated using established metrics such as BLEU (Bilingual Evaluation Understudy) scores. The BLEU score measures how closely the generated text matches a set of reference texts, usually human-generated captions. By comparing the BLEU scores of our model to those of state-of-the-art models, we can assess the relative quality and accuracy of our captions. High BLEU scores will indicate that the generated captions are both grammatically correct and contextually appropriate, providing meaningful descriptions of the images.
2. Classification Accuracy:
   The performance of the image classification model will be evaluated using standard metrics, including precision, recall, and F1-score. Precision measures the proportion of true positive predictions among all positive predictions, indicating the accuracy of the model in identifying relevant instances. Recall measures the proportion of true positive predictions among all actual positives, reflecting the model's ability to capture all relevant instances. The F1-score, which is the harmonic mean of precision and recall, provides a comprehensive measure of the model's accuracy. High scores in these metrics will demonstrate the model's effectiveness in accurately categorizing images into their respective classes.
3. User Studies:
   To assess the usability and effectiveness of the system in real-world scenarios, we plant to conduct user studies with visually impaired individuals. These studies will involve tasks where participants use the system to receive descriptions and categorizations of images. Feedback will be collected through surveys and interviews, focusing on the perceived usefulness, clarity, and accuracy of the system. Positive feedback from users will validate the practical applicability of the system and its potential to improve the accessibility of visual content for visually impaired individuals.
A successful outcome from the evaluation phase will result in useful categories generated by the clustering methodology. The system should be capable of accurately classifying new examples into the appropriate clusters, demonstrating adaptability and the ability to adjust over time. This adaptability will ensure the system remains effective and relevant as it encounters new data, contributing to its long-term utility and impact.
.
### 3.4	Proposed Timeline

Table 1. Proposed project timeline
Phase	Estimated Timing
Literature Review	3 days
Exploratory Data Analysis	1 week
Model Building & Training	1 week
Evaluation	1 week
Report writing & Presentation	1 week
Total Duration	~5 weeks

## 4.	SUCCESS CRITERIA
The success of this project will be determined by the following criteria:
1. Caption Generation Accuracy:
   The model's ability to generate accurate and meaningful captions will be evaluated using established metrics such as BLEU scores. Achieving competitive BLEU scores in comparison to existing state-of-the-art models will indicate the model's effectiveness in producing high-quality text descriptions of images. This criterion ensures that the generated captions are not only grammatically correct but also contextually relevant and informative, thereby meeting the high standards set by the leading models in the field.
2. Classification Performance:
   The image classification component of the model must demonstrate robust performance across several key metrics, including precision, recall, and F1-score. proficiency in accurately categorizing images.
3. User Satisfaction:
   An essential criterion for success is the positive feedback from user studies conducted with visually impaired individuals. These studies will assess whether the system provides meaningful and helpful descriptions and categorizations of images. User satisfaction will be measured through surveys and interviews, focusing on the perceived usefulness, clarity, and accuracy of the generated captions and classifications. Positive user feedback will validate the practical applicability of the model, ensuring that it meets the real-world needs of visually impaired users and enhances their interaction with visual content.
By meeting these success criteria, the project will demonstrate significant advancements in caption generation and image classification technologies, providing valuable tools for visually impaired individuals and contributing to the broader field of data science and machine learning.
## 5.	EVALUTATION
To evaluate the accuracy of generated captions, we employed the BLEU (Bilingual Evaluation Understudy) score, a well-established metric in natural language processing (NLP) for assessing text quality. The BLEU score quantifies the similarity between generated text and a set of reference texts, typically human translations or annotations. Scores range from 0 to 1, where a score of 1 indicates a perfect match and a score of 0 indicates no overlap.
Our approach involved comparing the generated captions against a random sample from the validation dataset using BLEU scores. This method is particularly effective when multiple reference examples are available, as it provides a more robust and comprehensive evaluation of the generated text. However, it is important to note that BLEU primarily measures surface-level similarity and may not fully account for semantic equivalence between the reference and predicted captions. Despite this limitation, BLEU remains a valuable tool for initial assessments of caption accuracy in NLP tasks.
The evaluation of the generated captions was conducted using the BLEU (Bilingual Evaluation Understudy) score, as mentioned, a widely recognized metric for assessing the quality of text generation models, particularly in tasks involving captioning. The BLEU score measures the similarity between the generated text and a set of reference texts, providing an objective means to evaluate the performance of the caption generation model.
BLEU Score Analysis
Upon evaluation, the generated captions achieved a BLEU score of 0.004 when compared against the test caption data. This score is significantly lower than expected, indicating that the generated captions are not closely matching the reference captions. The low BLEU score suggests that there is substantial room for improvement in the model's performance. This result points to the necessity for further tuning and diagnostic analysis to enhance the accuracy and relevance of the generated captions.
 
Example of generated caption from text image
Review of Caption Results
Despite the low BLEU score, a qualitative review of the generated captions reveals promising aspects. The accuracy score was 42%. Many of the generated captions capture key concepts and elements of the images, albeit not always in a precise manner that aligns with the reference captions. This observation indicates that while the model is capable of identifying and describing major components of the images, it struggles with fine-grained details and exact phrasing.
Several potential reasons for the low BLEU score include:
1. Vocabulary Mismatch:
   The generated captions may use different vocabulary than the reference captions, leading to lower BLEU scores despite conveying similar meanings.
2. Synonym and Paraphrasing Issues:
   The model may generate synonyms or paraphrased versions of the reference captions, which are semantically correct but not rewarded in BLEU score calculation due to the focus on n-gram overlap.
3. Structural Differences:
   Differences in sentence structure between generated and reference captions can also contribute to lower BLEU scores. The model might generate grammatically correct and meaningful sentences that differ syntactically from the reference captions.
Potential and Next Steps
The review of the caption results highlights the potential of the model to capture the essence of the images and the ability to cluster those captions and images into self-determined groups. Most concepts within the images seem to be recognized and described by the model, indicating a starting understanding of the image content. This suggests that with further tuning, the model's performance could be significantly improved.
To address the identified issues and improve the model's performance, the following steps are recommended:
1. Enhanced Tuning:
   Fine-tuning the model with a more diverse and extensive dataset could help it learn a broader range of vocabulary and sentence structures.
2. Diagnostic Analysis:
   Conducting a detailed diagnostic analysis to identify specific areas where the model fails, such as particular objects or scenes that are consistently misrepresented, can provide targeted insights for improvement.
3. Incorporation of Synonym Handling:
   Implementing techniques to account for synonyms and paraphrasing in the evaluation process can provide a more accurate assessment of the model's true performance.
4. User Feedback Integration:
   Incorporating feedback from user studies, particularly from visually impaired individuals, can provide practical insights into the usability and effectiveness of the generated captions, guiding further refinement of the model.
In conclusion, while the initial BLEU score evaluation indicates the need for substantial improvements, the qualitative review of the generated captions and the clustering results, with transformer embeddings and K-Means method, demonstrates the model's potential. By addressing the identified issues through targeted tuning and diagnostic efforts, the model's performance can be enhanced, leading to more accurate and meaningful image captions and classfication.
## 6.	CONCLUSION
The initial results from our study are promising, indicating that the methodologies and models applied have the potential to achieve the desired outcomes. However, the current models have not yet fully met all success criteria, suggesting that further iterations are necessary. The necessity for additional iterations highlights the complexity of the problem and the potential for further improvement in model performance through refinement and optimization.
We recommend progressing into a second iteration phase that focuses on model exploration and tuning. This phase will involve a deeper investigation into various model parameters and configurations, allowing for fine-tuning that could enhance the model’s accuracy and reliability. This iterative process is essential to adaptively improve the model, addressing any shortcomings identified during the initial phase and exploring new avenues for enhancement.
Moreover, our results show significant promise in using image feature extraction techniques to categorize data into self-identified text-based clusters. This finding underscores the potential of image feature extraction as a robust tool for data categorization, which can be particularly valuable in applications requiring the classification of images to simplify the experience for visually impaired persons. The ability to effectively categorize data using these techniques could lead to more efficient and accurate data analysis processes, ultimately contributing to advancements in the field.
In conclusion, while the initial outcomes are encouraging, the path to achieving full success criteria involves further iteration and refinement. The second iteration phase will be crucial for model enhancement, and the promising results of image feature extraction techniques open new possibilities for data categorization. These findings lay a strong foundation for future work and potential breakthroughs in image classification methodologies..
## 7.	REFERENCES
[1]	Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and Tell: A Neural Image Caption Generator. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3156-3164
[2]	Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929. 
[3]	Hodosh, Micah, Peter Young, and Julia Hockenmaier. "Framing image description as a ranking task: Data, models and evaluation metrics." Journal of Artificial Intelligence Research 47 (2013): 853-899
[4]	Karpathy, A., & Fei-Fei, L. (2015). Deep Visual-Semantic Alignments for Generating Image Descriptions. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3128-3137.
[5]	Gurari, D., Li, Q., Stangl, A., Guo, A., Lin, C., Grauman, K., ... & Bigham, J. P. (2020). VizWiz Grand Challenge: Answering Visual Questions from Blind People. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3608-3617.
[6]	Flick8k dataset: https://hockenmaier.cs.illinois.edu/8k-pictures.html. 

 

![image](https://github.com/aminteer/captioning_is_clustering/assets/30326111/d46d6141-2609-4615-993e-bf206758b3b8)
 ![image](https://github.com/aminteer/captioning_is_clustering/assets/30326111/9aca0e87-cb07-42e9-8381-e2e9c79f4f7e)
