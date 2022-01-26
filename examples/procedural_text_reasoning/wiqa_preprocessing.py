import argparse
import os
import logging
import torch
import jsonlines
import tqdm
import networkx as nx
import config

logger = logging.getLogger(__name__)

class WIQAProcessor:
    def __init__(self):
        self.name='chen zheng'
    """Processor for the WIQA data set."""

    def get_train_examples(self, data_dir, args):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self.create_examples(self.read_jsonl(os.path.join(data_dir, "train.jsonl")), "train", args)

    def get_dev_examples(self, data_dir, args):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self.create_examples(self.read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev", args)

    def get_test_examples(self, data_dir, args):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self.create_examples(self.read_jsonl(os.path.join(data_dir, "test.jsonl")), "test", args)

    def get_labels(self):
        return ["more", "less", "no_effect"]

    def read_jsonl(self, jsonl_file):
        lines = []
        print("loading examples from {0}".format(jsonl_file))
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines

    def create_examples(self, lines, set_type, args, add_consistency=True):
        """Creates examples for the training and dev sets."""
        examples = {}
        G = nx.Graph()
        # counter = 0

        # for (_, data_raw) in tqdm(enumerate(lines)):
        for data_raw in lines:
            # question = data_raw["question"]["stem"]
            # para_steps = " ".join(data_raw["question"]['para_steps'])
            # answer_labels = data_raw["question"]["answer_label"]
            # example_id = data_raw['metadata']['ques_id']
            # examples.append(
            #     InputExample(
            #         guid=example_id,
            #         text_a=question,
            #         text_b=para_steps,
            #         label=answer_labels))
            example_dict = {}

            ### extract question and doc information
            example_dict['question'] = data_raw["question"]["stem"]
            example_dict['ques_id'] = data_raw['metadata']['ques_id']
            example_dict['paragraph'] = " ".join(data_raw["question"]['para_steps'])
            # example_dict['paragraph_list'] = [p for p in data_raw["question"]['para_steps']]
            ### extract the multi-class labels
            if data_raw["question"]["answer_label"] == 'more':
                example_dict['more'] = 1
                example_dict['less'] = 0
                example_dict['no_effect'] = 0
            elif data_raw["question"]["answer_label"] == 'less':
                example_dict['more'] = 0
                example_dict['less'] = 1
                example_dict['no_effect'] = 0
            else:
                example_dict['more'] = 0
                example_dict['less'] = 0
                example_dict['no_effect'] = 1

            examples[example_dict['ques_id']] = example_dict

            ### networkx begin
            G.add_node(example_dict['ques_id']) ## the raw id include the _symmetric word or _transit word
            if '_symmetric' in example_dict['ques_id']: ## symmetric
                G.add_edge(example_dict['ques_id'].split('_symmetric')[0], example_dict['ques_id']) ## adding an edge between node2 and node1
            elif '@' in example_dict['ques_id']: ## transit a->b->c a->c
                G.add_edge(example_dict['ques_id'].split('@')[0], example_dict['ques_id'])
                G.add_edge(example_dict['ques_id'].split('@')[1].split('_transit')[0], example_dict['ques_id'].split('@')[0]) ## i add it, darius doesn't add it
                G.add_edge(example_dict['ques_id'].split('@')[1].split('_transit')[0], example_dict['ques_id'])
            ### networkx end

        ### networkx begin
        components = [i for i in nx.connected_components(G)]
        # print(components) {'influence_graph:823:748:15#1', 'influence_graph:823:748:15#1_symmetric_two'},
        components_size = [len(i) for i in nx.connected_components(G)]
        # print(components_size) ### [6, 2, 1, 53, 1, 4, 4, 1, 1, 2, 1, 1, 1, 1, 12]
        zipped_list = zip(components_size, components)
        sorted_pairs_reverse = sorted(zipped_list, reverse=True)
        tuples = zip(*sorted_pairs_reverse)
        components_size, components = [list(tuple) for tuple in tuples]
        print(components_size)
        for i in range(len(components)):
            components[i] = list(components[i])
            if components_size[i] >= args.batch_size or components_size[i] == 0:
                continue
            for j in range(i+1, len(components_size)):
                components[j] = list(components[j])
                if components_size[j] == 0 or components_size[j] + components_size[i] > args.batch_size\
                    or not examples[sorted(list(components[j]))[0]]['paragraph'] == \
                        examples[sorted(list(components[i]))[0]]['paragraph']:
                    continue
                # G.add_edge(sorted(list(components[j]))[0], sorted(list(components[i]))[0])
                G.add_edge(sorted(components[j])[0], sorted(components[i])[0])
                # print(components[i],components[j])
                components[i] = components[i] + components[j]
                components_size[j] = 0
                if components_size[i] >= args.batch_size:
                    break
        
        output_examples = []
        for nodes in sorted(list(nx.connected_components(G))):
            cur_list = sorted(list(nodes))
            if len(nodes) <= args.batch_size:
                output_examples.append(cur_list)
                continue
            for j in range(0, len(nodes), args.batch_size):
                output_examples.append(cur_list[j:j+args.batch_size])
        ### networkx end

        read_output = []
        for ques_id_list in output_examples:
            more_list = []
            less_list = []
            no_effect_list = []
            question_list = []
            para  = ''
            for ques_id in ques_id_list:
                cur_example = examples[ques_id]
                more_list.append(str(cur_example['more']))
                less_list.append(str(cur_example['less']))
                no_effect_list.append(str(cur_example['no_effect']))
                question_list.append(cur_example['question'])
                para = cur_example['paragraph']
            read_output.append(
                {
                    'ques_ids': '@@'.join(ques_id_list),
                    'question_list': '@@'.join(question_list),
                    'paragraph_intext': para,
                    'more_list': '@@'.join(more_list),
                    'less_list': '@@'.join(less_list),
                    'no_effect_list': '@@'.join(no_effect_list),
                }
            )

        return read_output

class InputFeatures:
    def __init__(self):
        self.name='chen zheng'

    def convert_qa_examples_to_features(self, args, dev=True, test=False):
        # Load data features from cache or dataset file
        if dev is True and test is False:
            cached_features_file = os.path.join(args.data_dir, 'cached_consistency_{}_{}_{}_{}'.format(
                'dev',
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length),
                str('wiqa'),
                ))
        elif dev is False and test is True:
            cached_features_file = os.path.join(args.data_dir, 'cached_consistency_{}_{}_{}_{}'.format(
                'test',
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length),
                str('wiqa'),
                ))
        elif dev is False and test is False:
            cached_features_file = os.path.join(args.data_dir, 'cached_consistency_{}_{}_{}_{}_{}'.format(
                'train',
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length),
                str('wiqa'), 
                args.random_ratio))
        else:
            raise NotImplementedError(
                "the mode should be either of train, dev or test.")

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            features = torch.load(cached_features_file)
        else:
            processor = WIQAProcessor()
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if dev is True and test is False:
                examples = processor.get_dev_examples(args.data_dir, args)
            elif dev is False and test is True:
                examples = processor.get_test_examples(args.data_dir, args)
            elif dev is False and test is False:
                examples = processor.get_train_examples(args.data_dir, args)
            else:
                raise NotImplementedError()
        return examples


def test_code():
    chen_arg = config.configuration()
    features = InputFeatures()
    dev_examples = features.convert_qa_examples_to_features(args=chen_arg)
    print(dev_examples[0:1])
            
test_code()

###



