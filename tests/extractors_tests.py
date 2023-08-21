import unittest
from extractors import extract_text_from_pdf
from extractors import extract_github_repo_info
from extractors import extract_data_from_csv
from extractors import extract_text_from_file
from extractors import extract_all_content_from_webpage


class TestPDFExtractor(unittest.TestCase):
    def test_extract_text_from_pdf(self):
        text = extract_text_from_pdf('example.pdf')
        print(text)
        self.assertIsNotNone(text)  # checks that some text was returned


class TestGitHubExtractor(unittest.TestCase):
    def test_extract_github_repo_info(self):
        info = extract_github_repo_info('https://github.com/openai/gpt-3')
        print(info)
        self.assertIsNotNone(info)  # checks that some info was returned


class TestCSVExtractor(unittest.TestCase):
    def test_extract_data_from_csv(self):
        data = extract_data_from_csv('test.csv')
        self.assertIsNotNone(data)  # checks that some data was returned


class TestTextExtractor(unittest.TestCase):
    def test_extract_text_from_file(self):
        text = extract_text_from_file('test.txt')
        self.assertIsNotNone(text)  # checks that some text was returned


class TestWebScraper(unittest.TestCase):
    def test_extract_all_content_from_webpage(self):
        content = extract_all_content_from_webpage('https://platform.openai.com/docs/api-reference')
        print(content)
        self.assertIsNotNone(content)  # checks that some content was returned


if __name__ == '__main__':
    unittest.main()
