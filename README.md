# Introduction

CNN-based small bowel lesion classification and XAI-based localization

- DL platform conversion (to Pytorch) and Code modularization
- Including experiments for master's thesis (Yunseob Hwang - Convolutional Neural Network Based Small Bowel Lesion Detection in Capsule Endoscopy, POSTECH, 2021)

# Thesis
<p align="center">
     <img alt="thesis" src="./images/thesis_summary.png"
          width=80% />
</p>
<br>

- Chapter 4

The MSG-GAN was utilized to synthesize fake SBCE samples (https://github.com/akanimax/BMSG-GAN)

<p align="center">
     <b> GAN training lapse </b><br>
     <img alt="GAN_training" src="./images/4_GAN_training_lapse.gif"
          width=70% />
</p>
<br>

<p align="center">
     <b> GAN synthesized samples </b><br>
     <img alt="GAN_result" src="./images/4_GAN_synthesized_samples.jpg"
          width=70% />
</p>



# *AI feedback Process
- AI model's inference can be used to collect training samples rapidly with reducing Experts' labeling cost 
- Eventually, it implies that this process could allow the AI model to be robust for itself
- Extended AI Research Fields: Active Learning, Online Learning, Continual Learning, AutoML...
- Related Work: Closing the Human-Data-AI Loop (https://url.kr/qikj9g 루닛)

<p align="center">
     <img alt="AI_feedback" src="./images/AI_feedback.png"
          width=80% />
</p>
<br>
