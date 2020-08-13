from ace05.reader import Reader
import config


def main():
    reader = Reader(config.path)
    for data_item in reader():
        text = data_item['text']
        refs = data_item['referables']
        rels = data_item['relations']
        events = data_item['events']
        # validate relations
        for rel_id, rel in rels.items():
            # relation has two arguments
            assert rel.arguments[0] is not None and rel.arguments[1] is not None
            # if there is rel.subtype, then rel.subtype is a rel.type
            assert not rel.subtype or rel.type in set(map(lambda r: r.dst, rel.subtype.is_a()))
            # 
        # print(data_item)
        pass
    print('done')


if __name__ == '__main__':
    main()
