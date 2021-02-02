import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from numpy import genfromtxt


def load_dicts(dataset):

    print("Loading class_entity, entity_class mappings from types file")
    if dataset == 'yago':
        type_file_path = "data/yago/yagoTransitiveType.tsv"
    else:
        type_file_path = "data/freebase/freebaseTypes.tsv"

    class_entity_dict = defaultdict(set)
    entity_class_dict = defaultdict(set)

    #read yago transitive types
    no = 0
    with open(type_file_path, "r") as fb_types:
        for line in fb_types:
            try:
                entity, cl = line.split()
                class_entity_dict[cl].add(entity)
                entity_class_dict[entity].add(cl)
                #print(class_entity_dict)
            except ValueError:
                continue
            if (no ==10):
                print(class_entity_dict)
                print(entity_class_dict)
                #break

    print(len(class_entity_dict.keys()))
    class_entity_dict_df = pd.DataFrame([(k, list(v)) for k, v in class_entity_dict.items()],columns=['entity', 'type'])

    #class_entity_dict_df = pd.DataFrame.from_records(class_entity_dict, columns=['entity', 'type'])
    #print(class_entity_dict_df.shape[0])
    #print(class_entity_dict_df.head(10))
    #class_entity_dict_df.to_csv("class_entity_dict_df_test.csv", sep="\t", header=None, index = False)


    entity_class_dict = pd.DataFrame.from_records(entity_class_dict, columns=['entity', 'type'])
    #print(entity_class_dict.shape[0])
    #print(entity_class_dict.head(10))
    #entity_class_dict.to_csv("entity_class_dict.csv", sep="\t", header=None )

    return class_entity_dict, entity_class_dict


def get_types(input_classes, class_entity_dict):

    entity_types = []
    for type in class_entity_dict.keys():
        #print (type, len(class_entity_dict[type]))
        #check if the current type is present in the list of input classes
        if type in input_classes:
            print (len(class_entity_dict[type]), type)
            #now loop through all entities stored for this class, and add to df with the class as type
            for entity in class_entity_dict[type]:
                #print(entity)
                entity_types.append([entity, type])

    return entity_types


def load_embeddings(entity_embeddings_df_unique, model_name, dataset):
    #  # Load the Model
    # Get the Pytorch Tensor Object for the entities
    # Put the labels into entity_list
    # Pick entities of interest from the pytorch sensor using entityIDS
    # Retrieve a numpy matrix from the pytorch tensor

    if dataset == "yago":
        path = '/embeddings/yago3-10-{0}.pt'.format(model_name)
    if dataset == "freebase":
        path = '/embeddings/fb15k-237-{0}.pt'.format(model_name)


    if "rdf2vec" in model_name:
        entities = []
        if dataset == "yago":
            path = '/embeddings/yago3-10-rdf2vec.tsv'
            entity_path = '/embeddings/yago3-10-rdf2vec-entities.tsv'
        if dataset == "freebase":
            path = '/embeddings/fb15k-237-rdf2vec.tsv'
            entity_path = '/embeddings/fb15k-237-rdf2vec-entities.tsv'

        embedding = genfromtxt(path, delimiter=',')
        if len(embedding) == 0:
            print("Reading RDF2Vec embedding is incorrect")
        with open(entity_path, "r") as rdf2vec_entity_file:
            for line in rdf2vec_entity_file:
                entities.append(line.replace("http://www.freebase.com","").replace("http://www.yago-knowledge.org/","").replace("\n",""))

        all_entity_string_list = entities


    else:

        print("Loading from ", path)
        checkpoint = load_checkpoint(path)
        model = KgeModel.create_from(checkpoint)

        train = torch.Tensor(range(0, model.dataset.num_entities())).long()
        #print (train)

        all_entity_string_list = model.dataset.entity_ids(train).tolist() #convert np array to list
        print("No of entities:", len(all_entity_string_list))
        print(all_entity_string_list[0])



    entities_strings = entity_embeddings_df_unique['entities'].tolist()
    print(entities_strings[0])

    print("Total entities", len(entities_strings)) # 936 , now just 199
    print("Unique entities", len(set(entities_strings)))

    print (entity_embeddings_df_unique.shape)
    type_freq = entity_embeddings_df_unique.groupby(['classes']).size().sort_values(ascending=False).reset_index(name='count')
    print (type_freq)
    #print(entities_strings)

    entities_types = entity_embeddings_df_unique.values.tolist()
    entity_embeddings = []
    print("Loading embeddings info from the model..")
    train_ids = []
    for row in entities_types:
        entity = row[0]
        type = row [1]
        if entity in all_entity_string_list: #if entity also in yago3 model
            idx = all_entity_string_list.index(entity)
            train_ids.append(idx)
            entity_embeddings.append([row[0],row[1]])
        #else:
        #    print (entity+ "not found")

    print(len(entity_embeddings))
    print(len(train_ids))


    #now obtain embeddings as well
    if "rdf2vec" in model_name:

        X = embedding[train_ids].tolist()
        #X= list(flatten(a))
        print(len(X))

    else:

        #get the vectors for trainingIDs from the pytorch tensor object
        train_id_tensor = torch.Tensor(train_ids).long()

        #get an embedding matrix for the classification/clustering
        X = model.get_s_embedder().embed(train_id_tensor).tolist()
        #print(X)


    entity_embeddings_df = pd.DataFrame(entity_embeddings, columns=['entities', 'classes'])

    #obtained the embeddings now, add them to df as well
    entity_embeddings_df['values'] = X
    # print (len(entity_embeddings_df))
    # print (entity_embeddings_df.head(1))
    #print(type(X))
    print("Embeddings length:", len(X))

    return entity_embeddings_df


def cluster_scores(X, cluster_labels, labels_true):

        labels_pred = cluster_labels

        ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
        print ("ARI:", round(ARI, 3))

        homogeneity_score = metrics.homogeneity_score(labels_true, labels_pred)
        print("homogeneity_score:", round(homogeneity_score, 3))

        completeness_score = metrics.completeness_score(labels_true, labels_pred)
        print("completeness_score:", round(completeness_score, 3))

        v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
        print("v_measure_score:", round(v_measure_score, 3))

        NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
        print("NMI:", round(NMI, 3))
        MI = metrics.mutual_info_score(labels_true, labels_pred)
        print("MI:", round(MI, 3))
        print("")


def cluster_entities(entity_embeddings_df):

        print("Clustering entities..")

        entity_embeddings_df['category'] = pd.factorize(entity_embeddings_df.classes)[0]
        #print ("Factorized class labels", diff_vectors_df.category)

        #find the unique classes again here for cluster number k
        classes_df = entity_embeddings_df.drop_duplicates(subset='classes', keep="first")

        #print ("Unique classes", classes_df.shape[0])
        #print (classes_df.classes)
        class_count = classes_df.shape[0]    #find the unique combos of s-o pairs

        X = entity_embeddings_df['values'].to_list()
        labels_true = entity_embeddings_df.category.tolist() #the actual class labels frm the df, first itn is same, rest merged


        print("Kmeans ..")
        kmeans = KMeans(n_clusters=class_count, random_state=0)
        clusters = kmeans.fit_predict(X)
        cluster_labels = kmeans.labels_
        cluster_scores(X, cluster_labels, labels_true)


        print ("Optics")
        clusters = OPTICS(min_samples=2).fit_predict(X)
        cluster_labels = clusters
        cluster_scores(X, cluster_labels, labels_true)


        print("Agglomerative")
        agglo = AgglomerativeClustering(n_clusters=class_count,linkage="ward", affinity='euclidean')
        clusters = agglo.fit_predict(X)
        cluster_labels = clusters
        cluster_scores(X, cluster_labels, labels_true)

        print("Spectral")
        sc = SpectralClustering(n_clusters=class_count, affinity='nearest_neighbors', n_neighbors= 10, n_init=100, assign_labels='kmeans')
        cluster_labels = sc.fit_predict(X)
        cluster_scores(X, cluster_labels, labels_true)



if __name__ == '__main__':

    #load the class names for experiments from files
    datasets = {'yago','freebase'}

    import collections
    experiments = collections.defaultdict(dict)

    experiments = {'yago':
        {
            'Level-1': ['wordnet_person_100007846', 'wordnet_organization_108008335', 'wordnet_body_of_water_109225146',
                        'wordnet_product_104007894'],
            'Level-2-Organizations': ['wordnet_musical_organization_108246613', 'wordnet_party_108256968',
                                      'wordnet_enterprise_108056231', 'wordnet_nongovernmental_organization_108009834'],
            'Level-2-Waterbodies': ['wordnet_stream_109448361', 'wordnet_lake_109328904', 'wordnet_ocean_109376198',
                                    'wordnet_bay_109215664', 'wordnet_sea_109426788'],
            'Level-2-Persons': ['wordnet_artist_109812338', 'wordnet_officeholder_110371450',
                                'wordnet_writer_110794014', 'wordnet_scientist_110560637',
                                'wordnet_politician_110450303'],
            'Level-3-Writers': ['wordnet_journalist_110224578', 'wordnet_poet_110444194', 'wordnet_novelist_110363573',
                                'wordnet_scriptwriter_110564905', 'wordnet_dramatist_110030277',
                                'wordnet_essayist_110064405', 'wordnet_biographer_109855433'],
            'Level-3-Scientists': ['wordnet_social_scientist_110619642', 'wordnet_biologist_109855630',
                                   'wordnet_physicist_110428004', 'wordnet_mathematician_110301261',
                                   'wordnet_chemist_109913824', 'wordnet_linguist_110264437',
                                   'wordnet_psychologist_110488865', 'wordnet_geologist_110127689',
                                   'wordnet_computer_scientist_109951070', 'wordnet_research_worker_110523076'],
            'level-3-Players': ['wordnet_football_player_110101634', 'wordnet_ballplayer_109835506',
                                'wordnet_soccer_player_110618342', 'wordnet_volleyball_player_110759047',
                                'wordnet_golfer_110136959'],
            'Level-3-Artists': ['wordnet_painter_110391653', 'wordnet_sculptor_110566072',
                                'wordnet_photographer_110426749', 'wordnet_illustrator_109812068',
                                'wordnet_printmaker_110475687']
        },

        'freebase':
            {
                'Level-1': ['wordnet_person_100007846', 'wordnet_organization_108008335',
                            'wordnet_body_of_water_109225146', 'wordnet_product_104007894'],
                'Level-2-Organizations': ['wordnet_musical_organization_108246613', 'wordnet_party_108256968',
                                          'wordnet_enterprise_108056231',
                                          'wordnet_nongovernmental_organization_108009834'],
                'Level-2-Persons': ['wordnet_artist_109812338', 'wordnet_officeholder_110371450',
                                    'wordnet_writer_110794014', 'wordnet_scientist_110560637',
                                    'wordnet_politician_110450303'],
                'Level-3-Artists': ['wordnet_painter_110391653', 'wordnet_sculptor_110566072',
                                    'wordnet_photographer_110426749', 'wordnet_illustrator_109812068',
                                    'wordnet_printmaker_110475687']
            }
    }



    for dataset in datasets:

        try:
            class_entity_dict, entity_class_dict = load_dicts(dataset)
        except:
            print("Error reading classes")
            continue

        print("Loaded dataset: {}".format(dataset))

        for class_set in experiments[dataset].keys():
            input_classes = experiments[dataset][class_set]
            print("")
            print("Looking at classes from level:", class_set, input_classes)
            # print ("Class found:", input_classes)
            print("Finding entities for these classes...")

            print("Now finding entities for classes in ", input_classes)
            entity_types = get_types(input_classes, class_entity_dict)

            entity_types_df = pd.DataFrame(entity_types, columns=['entities', 'classes'])
            print(entity_types_df.head(5))
            print(entity_types_df.shape)

            entity_types_df_unique = entity_types_df.drop_duplicates(subset=['entities'])
            print(entity_types_df_unique.shape)

            type_freq = entity_types_df_unique.groupby(['classes']).size().sort_values(ascending=False).reset_index(
                name='count')
            print(type_freq)

            model_names = {'complex', 'conve', 'distmult', 'transe', 'rescal', 'rdf2vec'}

            for model_name in model_names:
                print()
                print("Embedding model:", model_name)
                entity_embeddings_df = load_embeddings(entity_types_df, model_name, dataset)
                print(entity_embeddings_df.shape)

                print(entity_embeddings_df.head(5))
                print(entity_embeddings_df.shape)

                type_freq = entity_embeddings_df.groupby(['classes']).size().sort_values(ascending=False).reset_index(
                    name='count')
                print(type_freq)

                print()
                print("Removing repeats")
                entity_embeddings_df_unique = entity_embeddings_df.drop_duplicates(subset=['entities'])
                print(entity_embeddings_df_unique.shape)

                type_freq = entity_embeddings_df_unique.groupby(['classes']).size().sort_values(
                    ascending=False).reset_index(name='count')
                print(type_freq)

                cluster_entities(entity_embeddings_df_unique)





