import json
class Preprocessor(object):
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename

    def _read_generator(self):
        with open(self.filename, 'r', encoding='utf-8') as reader:
            for line in reader:
                yield line.strip()

    def process(self, content, **kwargs):
        """

        Parameters
        ----------
        content: str, 一篇语料文章.

        Returns
        -------
            list of tuples
        Examples:
                 [(src1, tgt1),
                  (src2, tgt2),
                  (src3, tgt3)...]
        """
        pass

    def input_fn(self, **kwargs):
        """

        Returns
        -------
            yield (src, tgt)
        """

        for line in self._read_generator():
            pairs = self.process(line, **kwargs)
            for pair in pairs:
                yield pair


class BaiDuPreprocessor(Preprocessor):
    def __init__(self, filename):
        super().__init__("百度语料预处理器", filename)

    def process(self, content, **kwargs):
        content = content.strip()
        pairs = []
        while content:
            # kwargs['max_len']=512  kwargs['threshold']=100
            # 例如 文档长1120：划分成512, 512+96
            # 因为 512+96  < 512+100
            if len(content) > kwargs['max_len'] + kwargs['threshold']:
                sub_content = content[0:kwargs['max_len']]
                content = content[kwargs['max_len']:]
            else:
                sub_content = content
                content = ""
            if len(sub_content) > kwargs['min_len']:
                pairs.append((sub_content[0: len(sub_content) // 2],sub_content[len(sub_content)//2:]))
        return pairs


# class ThucNewsPreprocessor(Preprocessor):
#     def __init__(self):
#         super().__init__("清华新闻语料预处理器")
#
#     def process(self, content):
#         pass
#

class News2016Preprocessor(Preprocessor):
    def __init__(self, filename):
        super().__init__("2016新闻语料预处理器", filename)

    def process(self, content, **kwargs):
        content = json.loads(content)['content'].strip()
        pairs = []
        while content:
            # kwargs['max_len']=512  kwargs['threshold']=100
            # 例如 文档长1120：划分成512, 512+96
            # 因为 512+96  < 512+100
            if len(content) > kwargs['max_len'] + kwargs['threshold']:
                sub_content = content[0:kwargs['max_len']]
                content = content[kwargs['max_len']:]
            else:
                sub_content = content
                content = ""
            if len(sub_content) > kwargs['min_len']:
                pairs.append((sub_content[0: len(sub_content) // 2], sub_content[len(sub_content)//2:]))
        return pairs


class WikiPreprocessor(Preprocessor):
    def __init__(self, filename):
        super().__init__("维基百科预处理器", filename)

    def process(self, content, **kwargs):
        content = json.loads(content)
        assert content['title'] == content['text'][0:len(content['title'])]
        content = content['text'][len(content['title']):].replace('\n', '').strip()
        pairs = []
        while content:
            # kwargs['max_len']=512  kwargs['threshold']=100
            # 例如 文档长1120：划分成512, 512+96
            # 因为 512+96  < 512+100
            if len(content) > kwargs['max_len'] + kwargs['threshold']:
                sub_content = content[0:kwargs['max_len']]
                content = content[kwargs['max_len']:]
            else:
                sub_content = content
                content = ""
            if len(sub_content) > kwargs['min_len']:
                pairs.append((sub_content[0: len(sub_content) // 2], sub_content[len(sub_content)//2:]))
        return pairs
