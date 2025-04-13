
import React from 'react';
import { ArrowDown, Github, FileText, ExternalLink } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

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
            <Button className="flex items-center gap-2">
              <Github size={18} />
              GitHub Repository
            </Button>
            <Button variant="outline" className="flex items-center gap-2">
              <FileText size={18} />
              Read the Paper
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
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-4">
              <p className="text-lg">
                NoMaD presents a diffusion-based approach to robot navigation that effectively 
                captures multimodal action distributions and generalizes to unseen environments. 
                Unlike traditional reinforcement learning methods, our approach learns directly 
                from expert demonstrations using goal-masked diffusion.
              </p>
              <p className="text-lg">
                The method employs a conditional diffusion model to generate action 
                sequences that guide the robot toward specified goals while avoiding obstacles and 
                adhering to environmental constraints.
              </p>
            </div>
            <div className="bg-muted rounded-lg p-6 h-80 flex items-center justify-center">
              <p className="text-muted-foreground text-center">System Architecture Diagram</p>
            </div>
          </div>
        </div>
      </section>

      {/* Results Section */}
      <section id="results" className="py-20 bg-muted">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-8 text-center">Results</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {[1, 2, 3].map((item) => (
              <Card key={item} className="overflow-hidden">
                <div className="h-48 bg-secondary flex items-center justify-center">
                  <p className="text-secondary-foreground">Result Visualization {item}</p>
                </div>
                <CardContent className="p-6">
                  <h3 className="text-xl font-semibold mb-2">Key Finding {item}</h3>
                  <p className="text-muted-foreground">
                    Description of important results and their implications for robot navigation systems.
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
          <div className="mt-12 p-6 bg-card rounded-lg border">
            <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
            <div className="h-80 bg-muted rounded flex items-center justify-center">
              <p className="text-muted-foreground text-center">Performance Chart</p>
            </div>
          </div>
        </div>
      </section>

      {/* Comparison Section */}
      <section id="comparison" className="py-20">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-8 text-center">Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted border-b">
                  <th className="p-4 text-left">Method</th>
                  <th className="p-4 text-left">Success Rate</th>
                  <th className="p-4 text-left">Path Efficiency</th>
                  <th className="p-4 text-left">Generalization</th>
                  <th className="p-4 text-left">Compute Requirements</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b hover:bg-muted/50">
                  <td className="p-4 font-semibold">NoMaD (Ours)</td>
                  <td className="p-4">95%</td>
                  <td className="p-4">High</td>
                  <td className="p-4">Excellent</td>
                  <td className="p-4">Moderate</td>
                </tr>
                <tr className="border-b hover:bg-muted/50">
                  <td className="p-4">Method A</td>
                  <td className="p-4">82%</td>
                  <td className="p-4">Medium</td>
                  <td className="p-4">Good</td>
                  <td className="p-4">Low</td>
                </tr>
                <tr className="border-b hover:bg-muted/50">
                  <td className="p-4">Method B</td>
                  <td className="p-4">78%</td>
                  <td className="p-4">Low</td>
                  <td className="p-4">Fair</td>
                  <td className="p-4">High</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="p-4">Method C</td>
                  <td className="p-4">90%</td>
                  <td className="p-4">High</td>
                  <td className="p-4">Poor</td>
                  <td className="p-4">Very High</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="mt-8 bg-muted p-6 rounded-lg">
            <h3 className="text-xl font-semibold mb-4">Key Advantages</h3>
            <ul className="list-disc pl-6 space-y-2">
              <li>Multimodal action distribution handling</li>
              <li>Robust generalization to unseen environments</li>
              <li>Sample efficiency during training</li>
              <li>Reduced reliance on explicit reward engineering</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References Section */}
      <section id="references" className="py-20 bg-muted">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-8 text-center">References</h2>
          <div className="space-y-6">
            {[1, 2, 3, 4].map((item) => (
              <div key={item} className="p-6 bg-card rounded-lg border flex justify-between items-start">
                <div>
                  <p className="font-semibold mb-2">[{item}] Author, A., Author, B., & Author, C. (2023)</p>
                  <p className="text-muted-foreground">"Title of the paper: Subtitle about robot navigation methods."</p>
                  <p className="text-sm text-muted-foreground mt-1">Journal of Robotics and Automation, 42(3), 256-270.</p>
                </div>
                <Button variant="ghost" size="sm" className="flex items-center gap-1">
                  <ExternalLink size={16} />
                  Link
                </Button>
              </div>
            ))}
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
