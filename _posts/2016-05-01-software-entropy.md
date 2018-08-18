---
layout: post
author: diogojc
title:  "Entropy in Sotware Development"
image: "/assets/2016-05-01-software-entropy/banner.jpg"
excerpt: "Thought on entropy in software development"
description: "Thought on entropy in software development"
date:   2016-05-01 21:38:52 +0200
categories: [software development]
tags: [software development, entropy]
---

The second law of thermodynamics is one of the most fundamental empirical natural laws in our universe. It states that entropy, in a closed system where energy is conserved, can only increase with time.

Entropy is a measure of how much things are mixed up. This mixed-up-ness implies some random process and is opposite to the concepts of symmetry and correlation which are the basis for structure.

Simply put, without an external source of energy, in the natural world, structures will always be mixed-up until they exist no more and cannot be unmixed back.

You can extrapolate this natural law on a lot of places, from the mundane, on why a car without maintenance will eventually rust and stop working, to the more metaphysical, on how life (structure) cannot start itself without external energy sources like the sun.

## Entropy in software development

I find it fun to try to extrapolate this concept to software development and to some of my work experience.

A software application organizes intent, design, technology and data into one of the infinitely possible configurations towards solving some given problem.

Economic, organizational and human interactions and behaviour are both highly unpredictable and the main drivers for change within software development.

Although there are possibly others, I argue, these are the biggest entropic forces and much like in the natural world they are always trying to raise entropy throughout time within your software application.

What does high entropy within a software application look like? If software is randomly changed in random directions it will more likely start outputting random behaviour and become harder to change.

Symptoms include slower time to market, fixing bad behaviour introduces more bad behaviour, hard to infer what the problem is looking at the solution… You start to see most of the characteristics of what’s called in software development lingo the “big ball of mud” anti-pattern.

The pervasiveness of this anti-pattern is a testament of the entropic forces hard at work in software development.

## Fighting entropy in software

This, does not mean that all projects will die the entropic death. It just means that something has to fight this natural law.

I believe it’s as impossible to remove the randomness of the biggest entropic forces in software development as is impossible to fight the second law of thermodynamics. The single best thing we can do to fight entropy is to make good design decisions, do them often and do them systemically.

A design decision is therefore leaving less to chance and a good design decision is leaving less to chance that the likelihood of unexpected behaviour and time needed to change increases.

Although the importance of design comes as no surprise, viewing design coldly as a tool to reduce entropy is, I think, a good thought exercise.

A bad design decision, introducing unneeded complexity, can actually further increase entropy. This is simply because the more structure or complexity you put in, the more energy you’re going to need in protecting it from entropy.

This also follows nicely Occam’s razor principle stating that, all things equal, when comparing ways to explain something, you should always pick the one the makes the least assumptions to things you don’t know yet.
