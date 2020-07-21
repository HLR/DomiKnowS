from ace05.reader import Reader
import config


def main():
    reader = Reader()
    for data_item in reader(config.path):
        print(data_item)
    print('done')


if __name__ == '__main__':
    main()
