import folia.main as folia

class TaggerFormat:
    """A simple format, one token per line, class in second column, empty line delimites sentences"""

    extension = "tagger"

    @staticmethod
    def from_txt(inputfilename: str, outputfilename: str, **kwargs):
        sentenceperline = kwargs.get('sentenceperline',False)
        wrotenewline = False
        with open(inputfilename,'r',encoding='utf-8') as fin:
            with open(outputfilename,'w',encoding='utf-8') as fout:
                for line in fin:
                    for token in line.split(" "):
                        wrotenewline = False
                        print(token,file=fout)

                    if (sentenceperline or not line.strip()) and not wrotenewline:
                        fout.write("\n")
                        wrotenewline = True #prevents two successive newlines



    def from_xml(inputfilename, outputfilename, **kwargs):
        doc = folia.Document(file=inputfilename)
        with open(outputfilename,'w',encoding='utf-8') as fout:
            for sentence in doc.sentences():
                for token in sentence.words():
                    print(token, file=fout)

            fout.write("\n")




