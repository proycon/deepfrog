import folia.main as folia

class TaggerFormat:
    """A simple format, one token per line, class in second column, empty line delimites sentences"""

    extension = "tagger"

    @staticmethod
    def from_txt(inputfilename: str, outputfilename: str, **kwargs):
        sentenceperline = kwargs.get('sentenceperline',False)
        wrotenewline = False
        linenr = 1
        tokennr = 1
        alignment = []
        with open(inputfilename,'r',encoding='utf-8') as fin:
            with open(outputfilename,'w',encoding='utf-8') as fout:
                for line in fin:
                    for token in line.split(" "):
                        wrotenewline = False
                        print(token,file=fout)
                        alignment.append((linenr,tokennr,None))
                        tokennr += 1

                    if (sentenceperline or not line.strip()) and not wrotenewline:
                        fout.write("\n")
                        linenr += 1
                        tokennr = 1
                        wrotenewline = True #prevents two successive newlines
        return alignment



    @staticmethod
    def from_xml(inputfile, outputfilename, **kwargs):
        alignment = []
        if isinstance(inputfile, folia.Document):
            doc = inputfile
        else:
            doc = folia.Document(file=inputfile)
        with open(outputfilename,'w',encoding='utf-8') as fout:
            for sentencenr, sentence in enumerate(doc.sentences()):
                for tokennr, token in enumerate(sentence.words()):
                    print(token, file=fout)
                    alignment.append((sentencenr,tokennr,token.id))
                fout.write("\n")
        return alignment

