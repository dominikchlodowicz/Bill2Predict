import unittest
from unittest import mock
import pathlib
import tempfile

from bill2predict.data import fetch_uci_dataset

class TestFetchUciDataset(unittest.TestCase):
    @mock.patch("bill2predict.data.fetch_uci_dataset.urllib.request.urlopen")
    @mock.patch("bill2predict.data.fetch_uci_dataset.shutil.copyfileobj")
    def test_download_calls_urlopen_and_copyfileobj(self, mock_copyfileobj, mock_urlopen):
        mock_resp = mock.Mock()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        dest = pathlib.Path(tempfile.gettempdir()) / "testfile.zip"
        dest.touch()

        fetch_uci_dataset._download("http://example.com/file.zip", dest)

        mock_urlopen.assert_called_once_with("http://example.com/file.zip")
        mock_copyfileobj.assert_called_once()

    @mock.patch("bill2predict.data.fetch_uci_dataset._download")
    @mock.patch("bill2predict.data.fetch_uci_dataset.zipfile.ZipFile.extractall")
    @mock.patch("bill2predict.data.fetch_uci_dataset.zipfile.ZipFile.__init__")
    def test_main_downloads_and_extracts(self, mock_zip_init, mock_extractall, mock_download):
        mock_zip_init.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            fetch_uci_dataset.main(["--output_dir", tmpdir])

            mock_download.assert_called_once()
            mock_zip_init.assert_called_once()

            expected_path = pathlib.Path(tmpdir).resolve()
            mock_extractall.assert_called_once_with(expected_path)

if __name__ == "__main__":
    unittest.main()
