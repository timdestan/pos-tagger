#PURPOSE

A Part of Speech tagger written in Ruby. I wrote this for a class project at UMD. As it does a fair amount of compute-bound work and is written in a slow dynamic language, I wouldn't recommend it for serious use.

#Requirements

Should work on any 1.9.X or greater version of Ruby.

#Usage

	ruby tagger.rb [OPTIONS]

To see command lines options:

	ruby tagger.rb --help

#About

There are two taggers, a Hidden Markov Model-based one, and a perceptron-based tagger. To choose a method use the
--method switch, e.g.:

	ruby tagger.rb --method PERCEPTRON

The training phase of the perceptron is rather long, so after a successful training run it (and the HMM,
for consistency) saves its state in JSON format to the data directory. To bypass the training in future
runs and read data from the JSON file instead (much, much, faster), use the --tagger-file switch:

	ruby tagger.rb --method Perceptron --tagger-file data/saved-perceptron-data.json

The program outputs all tags and words to a file in the data directory named by the method used to create it. The filename and location are not configureable (sorry!).

All data files, input and output, are stored in the data directory.

Other flags are explained in the program's help message.
