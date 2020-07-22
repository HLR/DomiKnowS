from ace05.reader import Reader
import config


def main():
    reader = Reader(config.path)
    for data_item in reader():
        # print(data_item)
        pass
    print('done')


if __name__ == '__main__':
    main()
