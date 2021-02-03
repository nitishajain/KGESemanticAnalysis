## Do Embeddings Actually Capture Knowledge Graph Semantics?

This is the repository for our research paper [Do Embeddings Actually Capture Knowledge Graph Semantics?](https://openreview.net/forum?id=vsxYOZoPvne) under review at the ESWC 2021 conference in research papers track.

### Abstract

Knowledge graph embeddings that generate vector space representations of knowledge graph triples, have gained considerable popularity in past years. Several embedding models have been proposed that achieve state-of-the-art performance for the task of triple completion in knowledge graphs. Relying on the presumed semantic capabilities of the learned embeddings, they have been leveraged for various other tasks such as entity typing, rule mining and conceptual clustering. However, a critical analysis of the utility as well as limitations of these embeddings for semantic representation of the underlying entities and relations has not been performed by previous work. In this paper, we performed a systematic evaluation of popular knowledge graph embedding models to obtain a better understanding of their semantic capabilities as compared to a non-embedding based approach. Our analysis brings attention to the fact that semantic representation in the knowledge graph embeddings is not universal, but restricted to a subset of the entities based on dataset characteristics. We provide further insights into the reasons for this behavior. The results of our experiments indicate that careful analysis of benefits of the embeddings needs to be performed when employing them for semantic tasks. 

### Code 

The classification and clustering code is self-sufficient, it will retrieve the embeddings of relevant entities and perform the semantic tasks on top of them. These tasks can be run independent of each other though they share the same data resources. 

### Data
The files with mapping of entities to their classes can be found in *data* for the different datasets. The yagoTransitiveType files is available for download directly from the Yago [web page](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads). 


### Results
Extended results, plots and figures accompanying the paper are available in *Supplementary Material* file.


### Pretrained Embedding Models
The KG embedding models that were used in this work were downloaded from https://github.com/uma-pi1/kge.git wherever available. The rest of the models were trained by us. The configuration files and the models are available for download from [here](https://owncloud.hpi.de/s/QIuLNwcaaInoMKo).
Once you have downloaded the models, make sure they are stored in a folder named *embeddings*. 
