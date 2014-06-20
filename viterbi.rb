#!/usr/bin/env ruby -wKU

require './logger'

class Fixnum
  # No integer is infinite.
  #
  def infinite?
    false
  end
end

# The trellis of probabilities.
#
class ViterbiTrellis
  include Logger
  # Construct trellis object for the given model.
  #
  def initialize(model)
    @model = model
    @memo = Hash.new()
    @bt = Hash.new()
    yield self if block_given?
  end
  # Finds max probability and state. Provided
  # block calculates the probability.
  #
  def find_max_prob_and_state
    max_state = nil
    max_prob = -Float::INFINITY
    @model.states.each do |st|
      probability = yield st
      if probability > max_prob
        max_prob = probability
        max_state = st
      end
    end
    return max_prob, max_state
  end
  # Transforms a P-value to log-space if needed.
  #
  def transform(p)
    return p
  end
  # Finds the sequence of states of maximum likelihood given the
  # provided sequence of events (observations). 
  # Returns the log probability as well as the "backtrace" which
  # contains the sequence of states.
  #
  def prob_and_bt(events)
    key = events
    # If we've already computed the probability of seeing this
    # chain of events, we can skip this whole mess and return it.
    unless @memo[key]
      case events.length
      when 1
        # Base case (first word in sentence)
        # Check all start probabilities. Find the max one.
        max_p, max_state = find_max_prob_and_state do |st|
          transform(@model.get_start_probability(st)) +
          transform(@model.get_emission_probability(events[0], st))
        end
        # Store it and done.
        @memo[key] = max_p
        @bt[key] = [max_state]
      else
        # Recursive case.
        *priors, event = *events
        # Find the probability and backtrace of the beginning.
        prob, bt = prob_and_bt(priors)
        if prob.infinite?
          # Impossible to get here. Set probability as infinite
          # and push another nil onto @bt (could probably just
          # leave it alone but we'll stay consistent.)
          @memo[key] = -Float::INFINITY
          @bt[key] = (bt.dup.push nil)
        else
          # Given that, find max likelihood state.
          max_p, max_state = find_max_prob_and_state do |st|
            transform(@model.get_transition_probability(bt.last, st)) +
            transform(@model.get_emission_probability(event, st))
          end
          @memo[key] = prob + max_p
          @bt[key] = (bt.dup.push max_state)
        end
      end
    end
    return @memo[key], @bt[key]
  end
end

# Viterbi trellis that applies logarithms at each step.
class LogarithmicViterbiTrellis < ViterbiTrellis
  # Transforms a P-value to log-space if needed.
  #
  def transform(p)
    return Math.log(p)
  end 
end