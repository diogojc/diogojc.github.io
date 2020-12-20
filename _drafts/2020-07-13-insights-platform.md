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
* Data
* Tools
* Solutions 

conclusions

----------

## Introduction
This article is about organizations becoming more "data-driven".

It is about how organizations can transform to systemically make decisions relying on data first and intuition second.

How can a highly diverse and segmented workforce, in an organization, make more and better decisions by combining their knowledge with access to data and technology?
How can the learned insights be put to work and reused across the organization?
What role should Information Technology (IT) departments play in this "data-driven" transformation?

This article will bring these together and propose a set of digital services through a *centralized platform open for decentralized contributions* I will refer to as Insights Platform (or simply Platform). I argue that, although not a guarantee, the services in this platform can help organizations in this transformation.

In a spectrum that begins in letting everyone 'do their thing' and ends in centralizing everything in IT, this article defends a position in the middle that allows individuals to quickly put their knowledge to work wherever they are in the organization while allowing IT to maintain safeguards, manage and bound the complexity of the underlying technology stack.

----------

## Why is this important?
Although the promises and pursue of a 'data-driven' transformation are not new in organizations [^1] [^2], in my experience, these organizations, and their partners, often struggle to organize the data and technology components into meaningful digital services that enable this transformation.

A great amount of focus is often given to technology patterns (e.g. data lakes, lambda/kappa architectures) and processes (e.g. Agile) but unfortunately, in my experience, these don't empower enough the organizations individuals and teams, and often end up as a simple rebrand of existing processes and artifacts.

I read once somewhere that good design is invisible. I think this applies here.
To elaborate on this I will outline the examples of bad design and how visible it becomes in this context.

As an employee, I understand a part of my organization business and I need to look at relevant data.
I quickly discover, that data is stored away in a silo protected by IT. After I build a business case and convince someone for access I get an SAP database export with terms and relationships I do not understand, I reach out to SAP admins to understand them. After four months I'm finally looking at data I can understand.

> ... data is stored away in a silo ... I get an SAP database export with terms and relationships I do not understand ... after four months I'm finally looking at data I can understand.

As an employee, I have relevant data but need tools (and compute) to analyze it.
I find through a colleague, they are using a tool that would fit my needs. Their setup took three months and spanned licensing, infrastructure, security controls and a lot of meetings with IT. I would have to go through the same, so I just use a trial version of the tool on my laptop.

> Their setup took three months and spanned licensing, infrastructure, security controls and a lot of meetings with IT ... I just use a trial version of the tool

As an employee I find a colleague in a homologous department in another country has created a solution that would be a great fit for my department. They have a team of developers, I need to staff, and reengineer my colleagues solution. I don't know what to do.

> I need to staff, and reengineer my colleagues solution. I don't know what to do.

These are real and recurring examples based on my experience in working with people in complex organizations.

Although this article will not give enough instructions to solve these problems, it can be used as a framework for implementation and minimize these problems.

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
Agile values for software development [^4] and it's frameworks (e.g Scrum, Kanban) have long crossed their origins to being applied to entire IT departments.

The trend of moving to software what can be achieved in software [^3] has put tremendous pressure in IT departments.
IT departments typically seen as a supporting area of the business, focusing on stability, predictability and well defined processes are now having to deal with a demand that changes often, fast and in some cases pushing what the industry has to offer.

Although IT can profit from Agile values and its frameworks, applying them top down without providing new services, to a changing demand, often ends in IT running the same processes using a slightly different vocabulary. Hardly a compelling return-on-investment.

----------

## Insights Platform digital services
*Overview, start with the individual needs, create services for that individual and x-functional teams, access to data, access to tools that fit his needs, ability to share outcomes and IP*


For individuals and teams to have a data based approach to decision making at least three needs must be met. Domain knowledge, data and technology.

The platform will require them to bring the domain knowledge they already have and will focus on serving the other two areas, data and the technology.

![Needs](/assets/2020-07-13-insights-platform/needs.png "cacaca asdas ")

These areas will be translated into three services resting in a common foundation.

![Services](/assets/2020-07-13-insights-platform/services.png)

The Data (as-a) service provides access to high value, integrated and interpretable datasets anywhere in the organization.

The Environments (as-a) service provides on-demand creation of environments with provisioned and configured technology for common workloads. The workloads should allow people to discover new insights, prototype ideas and create new solutions.

The Solutions (as-a) service provides sharing of the IP created in the platform with everyone in the organization through the form of managed multi-tenant solutions.

The foundation will be the result of the infrastructure, controls and processes created by the platform team to support the platform digital services. This should be the by-product of creating the services and not a goal in of itself.

----------

### Data
The data (as-a) service must deliver data using, as much as possible, terms and relationships directly captured from the language used by the workforce through an interface, or API, that is consumable by existing tools.

Because the language used to describe data is context specific so must our delivery of data. This service will consist of, possibly overlapping, sets of data, or datasets, that hold data in very narrow contexts. A context will be a specific slice of an arbitrary number of the organization sub-domains.
This is very much like Domain Driven Design approach to break down very large models filled with ambiguity and sometimes conflicting terminology to smaller domain models living in bounded contexts tackling narrower domain problems [^5].

![Bounded context](/assets/2020-07-13-insights-platform/bounded-context.png)

The argument for this approach is by having domain experts and data modelers share the same language in narrower sub-domains, it becomes easier for the experts to consume data from the produced models and for the modelers to work independently from each other in different domain areas.


At the center of this service will therefore be the implementation of these datasets. More concretely the models, the interfaces with the outside world and how to bring the entities arriving from those interfaces into our models.

Around these datasets will be the implementations of the interfaces to the outside world. These are the connections with transactional sources (pull/push, streaming/pooling, through a database or files, etc.) and the logic necessary to present this data through specific APIs (XMLA, SQL, Parquet, etc.)

This approach is quite similar to the ports and adapters pattern [^6] for isolating application logic and infrastructure. The motivation is to enable the development of the models to be decoupled from the ever evolving transactional systems and user tooling.

![Ports and adapters](/assets/2020-07-13-insights-platform/ports-and-adapters.png)

Typically the broader an organizations business model the more contexts there will be to fit data into, and therefore datasets that can delivered in this service. Prioritization should be done based on demand. Although the owners of this platform are always accountable, the implementation of specific datasets can be sourced to other parts of the organization closer to the context in question.

Although I argue the investment in integrating data and making it interpretable through the usage of context specific languages is this service biggest asset, it does not exclude the usage of tooling for metadata cataloguing helping users find the appropriate dataset location and gain trust how that data ends up in front of them.

On the topic of filling these models and interfaces with data the owners of the platform can take one of two approaches.
  * Connect to the systems of data as they are, copy the data and map it to the models.
  * Create an interface for ingestion and push the responsibility of connection, copying and mapping to other parts of the organization.
In terms of ports and adapters architecture, should the platform owners be responsible for only its ports or also its adapters?

Both approaches scale differently as you would expect for heterogeneous landscapes of data systems. 

<span style="color:red">**architect of this part goes here**</span>.


----------

### Environments

#### Goal: Deliver tools
The Tools (as-a) Service should provide quick access to (software) tools, that anyone in the organization can use to discover new insights and apply them in creating new solutions.

#### Concept of environments and project lifecycle
The access to tools happens through the concept of an environment. An environment is a place where tools are, or can be, provisioned, and which lifecycle is tied to a project. 
In this context a project simply means there is either time or money allocated for an individual or team to pursue the solution of a problem using data.
These environments must therefore be created quickly and on-demand by the workforce, and destroyed without affecting each other.

#### Content of environment
All environments should include non functionals like:
  * Access to a organization wide identity provider
  * Connectivity to relevant systems of data
  * Monitoring
  * Chargeback
  * Security controls

Available tools must meet only two requirements: i) can the tool be integrated with these foundational non-functionals (above) and ii) can professional support be offered through a (internally) unified support process.

Tools are made available to user in two ways: users can provision and configure this tooling by themselves or environments come with tools pre-provisioned and configured.

By pre-provisioning and configuring the tools in

#### Workloads and personas
Environment types can be created around the co-occurrence of 

Bellow examples of workloads I've seen often appear in multiple organizations:

Combining two datasets and applying a statistics python library to build a report identifying the factors can affect a chemical production.

Another example is a team of database, integration experts, modelers and analysts creating a monthly financial report.

Or even another one is a team of developers and machine learning experts creating a ordering forecasting solution.


#### Examples of workloads

#### Example lifecycle and template
<span style="color:red">**sequence diagram**</span>.



![Templates and environments](/assets/2020-07-13-insights-platform/temps-envs.png)


----------

### Solutions
*intellectual property reuse, managed services*
Collection of solutions, centrally managed in the platform by a global function, are ready to deliver insights on new data onboarded by a new OpCo.

For situations where the IP created in an environment becomes relevant enough for another team, the platform 


Isn't data as a service just another solution?
Why isn't everything a marketplace solution?

----------

![Environments and solutions](/assets/2020-07-13-insights-platform/envs-markt.png)

## Insights Platform management
*role of partners*

## Remarks and discussion
The platform as presented should read as a guide to organizing technology in meaningful services that help the workforce make use of data in their daily work.
The implementation and operation of such services are dependent on each organization size, structure, maturity and sourcing capabilities to name a few.

Companies in industries already disrupted by software will rely less on rationalization

In the end the services success relies on it's adoption.



[^1]: Data-Driven Transformation: Accelerate at Scale Now, Boston Consulting Group May 23<sup>rd</sup> 2017 [Link](https://www.bcg.com/publications/2017/digital-transformation-transformation-data-driven-transformation)

[^2]: Three keys to building a data-driven strategy, Mckinsey March 1<sup>st</sup> 2013 [Link](https://www.mckinsey.com/business-functions/mckinsey-digital/our-insights/three-keys-to-building-a-data-driven-strategy)

[^3]: Why Software Is Eating the World, Andreessen August 20<sup>th</sup> 2011 [Link](https://a16z.com/2011/08/20/why-software-is-eating-the-world/)

[^4]: Manifesto for Agile Software Development, 2001 [Link](http://agilemanifesto.org/iso/en/manifesto.html)

[^5]: Bounded Context, Martin Fowler, 2014 [Link](https://martinfowler.com/bliki/BoundedContext.html)

[^6]: Ports And Adapters Architecture, Alistair Cockburn 2006 [Link](http://wiki.c2.com/?PortsAndAdaptersArchitecture)

