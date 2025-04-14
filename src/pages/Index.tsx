import React from 'react';
import { ArrowDown, Github, FileText, ExternalLink, Cpu, Database, Layers, BarChart } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { ChartContainer } from "@/components/ui/chart";

const Index = () => {
  return (
  <div className="min-h-screen bg-background">
{/* Header/Navigation */}
<header className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm border-b border-border">
<div className="container mx-auto px-4 py-4 flex justify-between items-center">
<h1 className="text-xl font-bold">NoMaD</h1>
<nav>
<ul className="flex gap-6">
<li><a href="#introduction" className="hover:text-primary transition-colors">Introduction</a></li>
<li><a href="#results" className="hover:text-primary transition-colors">Results</a></li>
<li><a href="#comparison" className="hover:text-primary transition-colors">Comparison</a></li>
<li><a href="#references" className="hover:text-primary transition-colors">References</a></li>
</ul>
</nav>
</div>
</header>

{/* Hero Section */}
<section className="py-20 bg-gradient-to-b from-background to-muted">
<div className="container mx-auto px-4 text-center">
<h1 className="text-4xl md:text-5xl font-bold mb-6">NoMaD: Navigation with Goal-Masked Diffusion</h1>
<p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
A novel approach to robot navigation leveraging diffusion models with goal-masked learning.
</p>
<div className="flex flex-col sm:flex-row justify-center gap-4 mb-12">
<Button asChild className="flex items-center gap-2">
  <a href="https://github.com/ishobhnik/RobotNavigation" target="_blank" rel="noopener noreferrer">
    <Github size={18} />
    GitHub Repository
  </a>
</Button>
<Button asChild variant="outline" className="flex items-center gap-2">
  <a href="../#references">
    <FileText size={18} />
    Read the Paper
  </a>
</Button>
</div>
<a href="#introduction" className="inline-flex flex-col items-center text-muted-foreground hover:text-foreground transition-colors">
<span className="mb-2">Learn More</span>
<ArrowDown className="animate-bounce" />
</a>
</div>
</section>

{/* Introduction Section */}
<section id="introduction" className="py-20">
<div className="container mx-auto px-4">
<h2 className="text-3xl font-bold mb-8 text-center">Introduction</h2>

{/* Abstract */}
<div className="mb-12">
<h3 className="text-2xl font-semibold mb-4">Abstract</h3>
<div className="bg-card rounded-lg p-6 border">
<p className="text-lg leading-relaxed">
This report presents our work on implementing and analyzing the training and
perception components of a visual navigation system based on diffusion policies, as
adapted from the NOMAD (Navigation with Goal Masked Diffusion) framework. Our
approach combines a visual perception backbone based on EfficientNet and Transformer,
with a trajectory diffusion model for motion planning to support both goal-directed
and exploratory navigation. We trained our model on the SACSon, RECON
and go stanford dataset, which features diverse real-world trajectories across various
environments and robot platforms. We leverage a conditional diffusion model to
generate multimodal waypoint predictions, enabling the agent to reason about complex,
uncertain navigation scenarios. Our contributions include a detailed breakdown of the
model architecture, training methodology, and evaluation metrics—particularly focusing
on waypoint alignment through cosine similarity. We have left out the deployment aspects.
</p>
</div>
</div>

{/* Main Introduction */}
<div className="mb-12">
<h3 className="text-2xl font-semibold mb-4">Introduction</h3>
<p className="text-lg mb-6 leading-relaxed">
Robotic learning for navigation in unfamiliar environments requires the ability to perform
both task-oriented navigation (i.e., reaching a known goal) and task-agnostic exploration
(i.e., searching for a goal in a novel environment). Traditionally, these functionalities are
tackled by separate systems — for example, using subgoal proposals, explicit planning modules,
or distinct navigation strategies for exploration and goal-reaching.
</p>

{/* What is NoMaD */}
<div className="bg-muted p-6 rounded-lg mb-8">
<h4 className="text-xl font-semibold mb-4">What is NoMaD?</h4>
<p className="mb-4 leading-relaxed">
NoMaD is a transformer-based diffusion policy designed for long-horizon, memory-based
navigation, that can:
</p>
<ul className="list-disc pl-6 space-y-2 mb-4">
<li>Explore unknown places on its own (goal-agnostic behavior).</li>
<li>Go to a specific place or object when given a goal image (goal-directed behavior).</li>
</ul>
<p className="leading-relaxed">
Our project involves implementing the NoMaD Policy adapting its Transformer-based architecture
and conditional diffusion decoder to learn from a rich, multimodal dataset (SACSON, SCAND and RECON)
composed of real-world trajectories. Unlike traditional latent-variable models or methods that rely on
separate generative components for subgoal planning, the unified diffusion policy exhibits superior
generalization and robustness in unseen environments, while maintaining a compact model size.
In this report, we focus on the perception and training components of this policy, emphasizing how
a strong visual encoder combined with a diffusion-based decoder leads to improved alignment of predicted
and ground-truth waypoints. We analyze the training dynamics, present key quantitative metrics such as
cosine similarity and distance loss, and highlight the model's ability to generalize across diverse scenarios.
</p>
</div>
</div>

{/* Implementation Details */}
<div>
<h3 className="text-2xl font-semibold mb-6">Implementation Details</h3>

{/* Environment Setup */}
<div className="mb-8 bg-card rounded-lg p-6 border">
<div className="flex items-start gap-4">
<Cpu className="w-10 h-10 text-primary shrink-0 mt-1" />
<div>
<h4 className="text-xl font-semibold mb-3">Environment Setup</h4>
<p className="mb-2">
All experiments were conducted on a system equipped with an NVIDIA GeForce GTX 1660
SUPER GPU and a 12th Gen Intel® Core i9-12900K (24-core) processor.
</p>
<p className="mb-2">
The code was implemented in Python 3.12.9 in a conda environment. All the packages and
libraries used are listed in the requirements.txt file.
</p>
<p>
The codebase was adapted from an open-source repository, and experiment tracking was
performed using Weights & Biases (WandB) for logging loss curves, evaluation metrics,
and hyperparameter sweeps.
</p>
</div>
</div>
</div>

{/* Data Pipeline */}
<div className="mb-8 bg-card rounded-lg p-6 border">
<div className="flex items-start gap-4">
<Database className="w-10 h-10 text-primary shrink-0 mt-1" />
<div>
<h4 className="text-xl font-semibold mb-3">Data Pipeline</h4>
<p className="mb-2">
We trained the NoMaD model using the SACSoN and parts of RECON dataset, which
contains diverse real-world trajectories across various environments and robot platforms.
</p>
<p>
The SACSoN dataset was already processed and we it split into training and validation
sets using 80-20 split ratio. The RECON dataset consisted of bag files, which needed to be
preprocessed to extract RGB images, actions, and ground-truth waypoints.
</p>
</div>
</div>
</div>

{/* Training Procedure */}
<div className="bg-card rounded-lg p-6 border">
<div className="flex items-start gap-4">
<Layers className="w-10 h-10 text-primary shrink-0 mt-1" />
<div>
<h4 className="text-xl font-semibold mb-3">Training Procedure</h4>
<p className="mb-4">
Training was done on a single NVIDIA GPU using a batch size of 32 for 10 epochs.
</p>

<h5 className="text-lg font-medium mb-2">Training Objective:</h5>
<p className="mb-2">
We calculate the Mean Squares Error(MSE) between the predicted and the actual noise. For
overall loss function, we add the MSE temporal distance loss to the objective function.
</p>
<p className="mb-2 font-mono bg-muted p-2 rounded">
LN oM aD (ϕ, ψ, f, θ, fd ) = M SE(ϵk , ϵθ (ct , a0t + ϵk , k)) + λ.M SE(d(ot , og ), fd (ct ))
</p>
<p className="mb-4">where:</p>
<ul className="list-disc pl-6 space-y-1 mb-6">
<li>ψ, ϕ correspond to the visual encoders for the observation and goal images.</li>
<li>f corresponds to the transformer layers,</li>
<li>θ corresponds to diffusion process parameters,</li>
<li>fd corresponds to the temporal distance predictor.</li>
</ul>

<h5 className="text-lg font-medium mb-2">Training Configuration:</h5>
<ul className="list-disc pl-6 space-y-1">
<li>The weighting factor λ for the auxiliary distance loss was set to 10<sup>−4</sup>.</li>
<li>The model was trained using the AdamW optimizer with a learning rate of 1e<sup>−4</sup> and
a weight decay of 1e<sup>−2</sup>.</li>
<li>We applied cosine annealing to decay the learning rate over time.</li>
<li>We used goal masking with probability pm = 0.5, encouraging the model to generalize
across goal-visible and goal-agnostic contexts.</li>
<li>The diffusion process used a square cosine schedule with K = 10 denoising steps.</li>
<li>The ViNT observation encoder was an EfficientNet-B0, mapping 96 × 96 RGB images
into a 256-dimensional latent embedding.</li>
<li>The transformer used for context encoding had 4 layers and 4 attention heads.</li>
<li>Training checkpoints and EMA snapshots were saved at the end of each epoch for
model tracking and potential evaluation.</li>
<li>Training and validation loss curves were logged using Wandb for detailed inspection.</li>
</ul>
</div>
</div>
</div>
</div>
</div>
</section>

{/* Results Section */}
<section id="results" className="py-20 bg-muted">
<div className="container mx-auto px-4">
<h2 className="text-3xl font-bold mb-8 text-center">Results</h2>

{/* Goal Conditioned Evaluation */}
<div className="mb-12">
<div className="bg-card rounded-lg p-6 border mb-6">
<h3 className="text-2xl font-semibold mb-4">Goal Conditioned (GC) Evaluation</h3>
<p className="mb-6">
We set the goal mask m = 0 to evaluate the model's performance in goal-directed navigation.
</p>

{/* Distance Loss */}
<div className="mb-10">
<h4 className="text-xl font-medium mb-4">Distance Loss</h4>
<div className="grid md:grid-cols-2 gap-6 mb-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_distloss_nomad.png" alt="Distance Loss Plot 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_dist_loss_test.png" alt="Distance Loss Plot 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<p>
We see that the loss has plateaued after the first epoch, both in test and training sets,
indicating the model has learned to predict the distance to the goal.
</p>
</div>

{/* Action Loss */}
<div className="mb-10">
<h4 className="text-xl font-medium mb-4">Action Loss</h4>
<div className="grid md:grid-cols-2 gap-6 mb-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_actionloss_nomad.png" alt="Action Loss Plot 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_actionloss_test_nomad.png" alt="Action Loss Plot 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<p className="text-sm text-center text-muted-foreground mt-2">
Figure 3: Action loss comparison between training and validation sets under goal-conditioned evaluation.
</p>
</div>

{/* GC Action Waypts Cos Sim */}
<div className="mb-10">
<h4 className="text-xl font-medium mb-4">GC Action Waypts Cos Sim</h4>
<div className="grid md:grid-cols-2 gap-6 mb-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_action_waypts_cos_sim_train.png" alt="GC Action Waypts Plot 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_action_waypts_cos_sim_test.png" alt="GC Action Waypts Plot 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<p className="text-sm text-center text-muted-foreground mt-2">
Figure 4: Cosine similarity between predicted and ground-truth waypoints on the training and validation sets under goal-conditioned evaluation.
</p>
</div>

{/* GC Multi Action Waypts Cos Sim */}
<div className="mb-10">
<h4 className="text-xl font-medium mb-4">GC Multi Action Waypts Cos Sim</h4>
<div className="grid md:grid-cols-2 gap-6 mb-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_multi_action_waypts_cos_sim.png" alt="GC Multi Action Waypts Plot 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/gc_multi_action_waypts_cos_sim_test.png" alt="GC Multi Action Waypts Plot 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<p className="text-sm text-center text-muted-foreground mt-2">
Figure 5: Cosine similarity between predicted and ground-truth multi-waypoints on the training and validation sets under goal-conditioned evaluation.
</p>
</div>
</div>

{/* Additional Metrics */}
<div className="grid md:grid-cols-2 gap-6 mb-10">
{/* Distance Loss */}
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">Distance Loss</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/dist_loss.png" alt="Additional Distance Loss Plot" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>

{/* Learning Rate */}
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">Learning Rate</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/lr.png" alt="Learning Rate Plot" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
</div>

{/* Diffusion Loss */}
<div className="bg-card rounded-lg p-6 border mb-10">
<h4 className="text-xl font-medium mb-4">Diffusion Loss</h4>
<div className="grid md:grid-cols-2 gap-6 mb-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/diffusion_loss.png" alt="Diffusion Loss Plot 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/diffusion_eval_loss_goal_masking.png" alt="Diffusion Loss Plot 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/diffusion_eval_loss_random_masking.png" alt="Diffusion Loss Plot 3" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/diffusion_eval_loss_no_masking.png" alt="Diffusion Loss Plot 4" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<p className="text-sm text-center text-muted-foreground mt-2">
Figure 8: Diffusion loss on training set and evaluation loss with different masking strategies
</p>
</div>

{/* Total Loss */}
<div className="bg-card rounded-lg p-6 border mb-10">
<h4 className="text-xl font-medium mb-4">Total Loss</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={21/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/total_loss.png" alt="Total Loss Plot" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>

{/* Action Samples */}
<div className="grid md:grid-cols-2 gap-6">
{/* Train Action Samples */}
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">Train Action Samples</h4>
<div className="grid grid-cols-2 gap-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/train_action_samples_1.png" alt="Train Sample 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/train_action_samples_2.png" alt="Train Sample 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/train_action_samples_3.png" alt="Train Sample 3" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/train_action_samples_4.png" alt="Train Sample 4" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
</div>

{/* SACSON Test Action Samples */}
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">SACSON Test Action Samples</h4>
<div className="grid grid-cols-3 gap-4">
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/sacson_test_action_samples_1.png" alt="Test Sample 1" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/sacson_test_action_samples_2.png" alt="Test Sample 2" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={1}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/sacson_test_action_samples_3.png" alt="Test Sample 3" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
</div>
</div>
</div>
</div>
</section>

{/* Comparison Section */}
<section id="comparison" className="py-20">
<div className="container mx-auto px-4">
<h2 className="text-3xl font-bold mb-8 text-center">Comparison</h2>

<div className="bg-card rounded-lg p-6 border mb-10">
<h3 className="text-2xl font-semibold mb-6">Comparison with ViNT</h3>
<div className="grid md:grid-cols-2 gap-8 items-center mb-8">
<div>
<p className="mb-4 leading-relaxed">
To establish a baseline, we trained the original ViNT architecture using the same dataset
and training hyperparameters as used for NoMaD.
</p>
<p className="mb-4 leading-relaxed">
We used the same EfficientNet-B0 encoder and transformer architecture, but without the
diffusion decoder.
</p>
<p className="mb-4 leading-relaxed">
We evaluated both models under identical conditions on the validation set, using metrics
such as Mean Squared Error (MSE) for actions and cosine similarity between predicted and
ground-truth waypoints.
</p>
<p className="leading-relaxed">
Interestingly, we observed that in the goal-conditioned (GC) setting, both ViNT and
NoMaD achieved comparable performance across key metrics. This suggests that the introduction
of a diffusion-based decoder in NoMaD does not degrade performance in goal-directed
planning tasks. Instead, it provides a more expressive generative mechanism while maintaining
task performance.
</p>
</div>
</div>

<p className="mb-6 leading-relaxed">
This supports the hypothesis that diffusion-based modeling can match deterministic
approaches in accuracy while offering better generative flexibility, particularly for
multi-modal or long-horizon planning scenarios.
</p>

{/* Training Metrics */}
<div className="mb-8">
<h4 className="text-xl font-medium mb-4">Training Metrics</h4>
<ul className="list-disc pl-6 space-y-2">
<li>Final training loss: 0.003</li>
<li>Cosine similarity: 0.47 (multi-action waypoints)</li>
<li>Distance loss: 10.2</li>
</ul>
</div>

{/* Observations */}
<div>
<h4 className="text-xl font-medium mb-4">Observations</h4>
<p className="leading-relaxed">
Loss plateaued after around 5,000 batches. Training logs show improvement in cosine similarity
and reduction in loss. Action losses remained stable across UC and GC branches.
</p>
</div>
</div>
{/* More visual comparisons */}
<div className="grid md:grid-cols-2 gap-6 mt-6">
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">NoMaD Distance Loss</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/nomad_dist_loss.jpg" alt="NoMaD Distance Loss" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">ViNT Distance Loss</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/vint_dist_loss.jpg" alt="ViNT Distance Loss" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
</div>

<div className="grid md:grid-cols-2 gap-6 mt-6">
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">NoMaD Action Loss</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/nomad_action_loss.jpg" alt="NoMaD Action Loss" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
<div className="bg-card rounded-lg p-6 border">
<h4 className="text-xl font-medium mb-4">ViNT Action Loss</h4>
<div className="bg-muted rounded-lg overflow-hidden">
<AspectRatio ratio={16/9}>
<div className="h-full w-full flex items-center justify-center">
<img src="/images/vint_action_loss.jpg" alt="ViNT Action Loss" className="w-full h-full object-contain" />
</div>
</AspectRatio>
</div>
</div>
</div>


</div>
</section>

{/* References Section */}
<section id="references" className="py-20 bg-muted">
  <div className="container mx-auto px-4">
    <h2 className="text-3xl font-bold mb-8 text-center">References</h2>
    <div className="space-y-6">

      {/* Reference 1 */}
      <div className="p-6 bg-card rounded-lg border flex justify-between items-start">
        <div>
          <p className="font-semibold mb-2">NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration</p>
          <p className="text-sm text-muted-foreground">arXiv:2310.07896</p>
        </div>
        <Button variant="ghost" size="sm" className="flex items-center gap-1">
          <ExternalLink size={16} />
          <a href="https://arxiv.org/pdf/2310.07896" target="_blank" rel="noopener noreferrer">Link</a>
        </Button>
      </div>

      {/* Reference 2 */}
      <div className="p-6 bg-card rounded-lg border flex justify-between items-start">
        <div>
          <p className="font-semibold mb-2">ViNT: A Foundation Model for Mobile Robotics</p>
          <p className="text-sm text-muted-foreground">arXiv:2306.14846</p>
        </div>
        <Button variant="ghost" size="sm" className="flex items-center gap-1">
          <ExternalLink size={16} />
          <a href="https://arxiv.org/abs/2306.14846" target="_blank" rel="noopener noreferrer">Link</a>
        </Button>
      </div>

      {/* Reference 3 */}
      <div className="p-6 bg-card rounded-lg border flex justify-between items-start">
        <div>
          <p className="font-semibold mb-2">Vision Transformers (ViT)</p>
          <p className="text-sm text-muted-foreground">arXiv:2010.11929</p>
        </div>
        <Button variant="ghost" size="sm" className="flex items-center gap-1">
          <ExternalLink size={16} />
          <a href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener noreferrer">Link</a>
        </Button>
      </div>

      {/* Reference 4 */}
      <div className="p-6 bg-card rounded-lg border flex justify-between items-start">
        <div>
          <p className="font-semibold mb-2">EfficientNet</p>
          <p className="text-sm text-muted-foreground">arXiv:1905.11946</p>
        </div>
        <Button variant="ghost" size="sm" className="flex items-center gap-1">
          <ExternalLink size={16} />
          <a href="https://arxiv.org/abs/1905.11946" target="_blank" rel="noopener noreferrer">Link</a>
        </Button>
      </div>

      {/* Reference 5 */}
      <div className="p-6 bg-card rounded-lg border flex justify-between items-start">
        <div>
          <p className="font-semibold mb-2">DiffusionPolicy: Visuomotor Policy Learning via Action Diffusion</p>
          <p className="text-sm text-muted-foreground">arXiv:2303.04137</p>
        </div>
        <Button variant="ghost" size="sm" className="flex items-center gap-1">
          <ExternalLink size={16} />
          <a href="https://arxiv.org/abs/2303.04137" target="_blank" rel="noopener noreferrer">Link</a>
        </Button>
      </div>

      {/* Reference 6 */}
      <div className="p-6 bg-card rounded-lg border flex justify-between items-start">
        <div>
          <p className="font-semibold mb-2">visualnav-transformer (GitHub Codebase)</p>
          <p className="text-sm text-muted-foreground">Used as baseline implementation for ViNT</p>
        </div>
        <Button variant="ghost" size="sm" className="flex items-center gap-1">
          <ExternalLink size={16} />
          <a href="https://github.com/robodhruv/visualnav-transformer.git" target="_blank" rel="noopener noreferrer">Link</a>
        </Button>
      </div>

    </div>
  </div>
</section>

{/* Footer */}
<footer className="py-10 border-t">
<div className="container mx-auto px-4 text-center">
<p className="text-muted-foreground">© 2025 NoMaD Research Team. All rights reserved.</p>
<div className="flex justify-center gap-6 mt-4">
<a href="#" className="hover:text-primary transition-colors">GitHub</a>
<a href="#" className="hover:text-primary transition-colors">Contact</a>
<a href="#" className="hover:text-primary transition-colors">Privacy Policy</a>
</div>
</div>
</footer>
</div>
);
};

export default Index;