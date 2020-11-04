---
layout: default
author: diogojc
title:  "Insights platform for 'data-driven' organizations"
image: "/assets/2018-09-08-artistic-style/images/thumbnail2.png"
excerpt: "Quote here."
description: "Summary here."
date:   2020-10-29
categories: [enterprise]
tags: [data, transformation, insights, platform, IT, organization, services, analytics, lake, warehouse, embedded]
---

## Executive summary
Data driven transformation and IT role
Organizations struggle to organize technology into meaningful digital services
Centralized platform open to decentralized contributions proposition
Digital services
    Data
    Tools
    Solutions 
conclusions

----------

## Introduction
This article is about organizations becoming more "data-driven". 

How can a highly diverse and segmented workforce, in an organization, make more and better decisions by combining their knowledge with access to data and technology?

How can the learned insights be put to work and reused across the organization?

Last, but not least, what role should Information Technology (IT) departments play in this "data-driven" transformation?

This article will bring these together and propose a set of digital services through a centralized platform open for decentralized contributions, that I will refer to as Insights Platform.

In a spectrum that begins in letting everyone 'do their thing' and ends in centralizing everything in IT, this article defends a position in the middle that allows individuals to put their knowledge to work wherever they are in the organization while allowing IT to maintain safeguards, manage and bound the underlying technology stack.

----------

## Why is this important?
Although the promises and pursue of a 'data-driven' transformation are not new in organizations [^1] [^2], in my experience, these organizations, and their partners, often struggle to organize the data and technology components into meaningful digital services that enable this transformation.

A great amount of focus is often given to technology patterns (e.g. data lakes, lambda/kappa architectures) and processes (e.g. Agile) but unfortunately, in my experience, these don't empower enough the organizations individuals and teams, and often end up as a simple rebrand of existing processes and artifacts.

Somewhere I read good design is invisible. I think this applies here.
To elaborate on this I will outline the examples of bad design in this context.

As an employee, I understand a part of my organization business and I need to look at relevant data.
I quickly discover, that data is stored away in a silo protected by IT. After I build a business case and convince my access can deliver something valuable I get access to a extract of a SAP database with terms and relationships I do not understand. I reach out to SAP database admins to understand how to interpret the data. After four months in and I'm finally looking at data that I understand.

> ... data is stored away in a silo ... I get access to a extract of a SAP database with terms and relationships I do not understand ... four months in and I'm finally looking at data that I understand

As an employee, I have relevant data but need tools (and compute) to analyze it.
I find through a colleague, they are using a tool that would fit my needs. Their setup took three months and spanned licensing, infrastructure, security controls and a lot of meetings with IT. I would have to go through the same, so I just use a trial version of the tool.

> Their setup took three months and spanned licensing, infrastructure, security controls and a lot of meetings with IT ... I just use a trial version of the tool.

As an employee I find a colleague in a homologous department in another country has created a solution that would be a great fit for my department. They have a team of developers, I need to staff, and reengineer my colleagues solution. I don't know what to do.

> I need to staff, and reengineer my colleagues solution. I don't know what to do.

These are real and recurring examples based on my experience in working with people in complex organizations.

I will not claim that this article will give enough instruction that will remove all these problems, but believe it can be used as a framework to minimize them to some extent.

The platform should strive to remove these hurdles and hide away the complexity of bringing data, tools and intellectual property to the employee.

----------

## Related topics

### Data Warehousing and Business Intelligence
Almost every organization has one or multiple data warehouses of some sort. Whenever you have one or more systems where data comes into existence and someone wants to perform aggregations on that data, you will find one.

The biggest consumers of this are people in front of a Business Intelligence (BI) tool connected to the Data Warehouse. They are typically looking back in time on different aspects of the organization (e.g. financial, orders, incidents, customer, etc).

Both the Data Warehouse and BI tool working together allow someone who understands nothing of the underlying complexity in the organization data and systems to be able to create, view and share, among other tings, the revenue of the organization grouped by its products.

Although their prevalence and longevity are a testament to its success recently cracks appeared, the first is they struggle to service users with increasing technical and analytics skills looking for more then reporting. The other one is because it treats all data as having the same value its costs tend to scale exponentially.

### Data Lake(s)
The acumen from the Hadoop ecosystem showed that cheap compute and storage could tackle the explosion of data growth. Vendors quickly jumped to monetize this new trend under the moniker of Data Lake with the motto "hoard all the data and the users will come". 

Even though the term Data Lake is so fuzzy to the point of meaninglessness, its motto combined with low upfront costs proved very attractive for IT managers. Enterprise architects also jumped in the latest tech trend and data lake adoption spread across the industry.

As unaltered copies of data landed in Data Lakes, end users were left scratching their heads on how to interpret the data and left, turning the lake into a graveyard. 
IT workers struggled with applying security controls on files creating copies of copies to perform things like row-level security, turning the lake into a swamp.

Some organizations soon after realized their data lake would make more sense when encapsulated by their Data Warehouse and have reaped some of it's benefits, namely how to deal, in a cost effective way, an increasing amount of data.

### Embedded Analytics
The idea to bring analytics capabilities to the employees existing tools and processes is called embedded analytics.

This bottom-up approach appeal is simple, pay a bit more for whatever software your employees are already using (e.g. Customer Relationship Management (CRM)) and overnight add analytical powers to their existing flows and processes.

Although simple this approach does deliver on it's promises and currently any serious software vendor in the enterprise has to offer this to remain competitive.

The danger with this approach is, if the organization only relies on this approach, their data strategy will, for the better or worse, soon be completely coupled to it's vendors. 
Although this might be fine in supporting areas of the business, for the business areas creating a competitive advantage the risk exposure can simply become too high. 
For example: "What if the vendor becomes too expensive? What if we need other capabilities? What if they go bankrupt?"


### Agile IT
Agile values for software development and it's frameworks (e.g Scrum, Kanban) have long crossed their origins to being applied to entire IT departments.
If you believe every organization will become a
 
For organizations where IT has typically been seen as a supporting role

----------

## Insights Platform digital services
*Overview, start with the individual needs, create services for that individual and x-functional teams, access to data, access to tools that fit his needs, ability to share outcomes and IP*

### Data
*Domains, Models, Problem and solution spaces*

### Tools and Environments
*self-service in a managed environment, acceleration of procurement and approval of tooling, rationalization of costs, standardization of support*

### Marketplace
*intellectual property reuse, managed services*


## Insights Platform management
*role of partners*

## Remarks and discussion
Here go the remarks




[^1]: Data-Driven Transformation: Accelerate at Scale Now, Boston Consulting Group May 23<sup>rd</sup> 2017 [Link](https://www.bcg.com/publications/2017/digital-transformation-transformation-data-driven-transformation)

[^2]: Three keys to building a data-driven strategy, Mckinsey March 1<sup>st</sup> 2013 [Link](https://www.mckinsey.com/business-functions/mckinsey-digital/our-insights/three-keys-to-building-a-data-driven-strategy)


[^3]: Why Software Is Eating the World, Andreessen August 20<sup>th</sup> 2011 [Link](https://a16z.com/2011/08/20/why-software-is-eating-the-world/)

