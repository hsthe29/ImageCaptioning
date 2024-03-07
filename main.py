from sentencepiece import SentencePieceProcessor
from image_captioning.dataset import load_data, FlickrDataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sp = SentencePieceProcessor()
    sp.Load("tokenizer/english.model")
    print(sp.EncodeAsPieces("Jang Judy"))
    print(sp.Encode(["Jang Judy", "the"], add_bos=True, add_eos=True))
    
    train_df, dev_df, test_df = load_data()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
