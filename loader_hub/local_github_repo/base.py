"""
Github repository reader.

Retrieves the contents of a Github repository and returns a list of documents.
The documents are either the contents of the files in the repository or
the text extracted from the files using the parser.
"""

import logging
import os
import pathlib
import tempfile
from typing import List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR
from llama_index.readers.file.base_parser import ImageParserOutput
from llama_index.readers.schema.base import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocalGithubRepositoryReader")

def print_if_verbose(verbose: bool, message: str) -> None:
    """Log message if verbose is True."""
    if verbose:
        print(message)


def get_file_extension(filename: str) -> str:
    """Get file extension."""
    return f".{os.path.splitext(filename)[1][1:].lower()}"

class LocalGithubRepositoryReader(BaseReader):
    def __init__(
        self,
        local_repo_path: str,
        use_parser: bool = True,
        verbose: bool = False,
        ignore_file_extensions: Optional[List[str]] = None,
        ignore_directories: Optional[List[str]] = None,
    ):
        super().__init__()
        self._local_repo_path = local_repo_path
        self._use_parser = use_parser
        self._verbose = verbose
        self._ignore_file_extensions = ignore_file_extensions
        self._ignore_directories = ignore_directories

    def load_data(self) -> List[Document]:
      documents = []

      print_if_verbose(self._verbose, f"ignore_directories {self._ignore_directories}")
      print_if_verbose(self._verbose, f"ignore_file_extensions {self._ignore_file_extensions}")
      for root, dirs, files in os.walk(self._local_repo_path):
          rel_root = os.path.relpath(root, self._local_repo_path)
          path_parts = pathlib.Path(rel_root).parts

          if self._ignore_directories is not None:
            dirs[:] = [d for d in dirs if d not in self._ignore_directories]

          for file in files:
              file_path = os.path.join(root, file)
              rel_path = os.path.relpath(file_path, self._local_repo_path)

              if self._ignore_file_extensions is not None:
                  if get_file_extension(file_path) in self._ignore_file_extensions:
                      print_if_verbose(
                          self._verbose,
                          f"ignoring file {file_path} due to file extension",
                      )
                      continue

              with open(file_path, "rb") as f:
                  file_content = f.read()

              if self._use_parser:
                  document = self._parse_supported_file(
                      file_path=rel_path,
                      file_content=file_content,
                      tree_sha=None,
                      tree_path=rel_path,
                  )
                  if document is not None:
                      documents.append(document)
                  else:
                      continue

              try:
                  decoded_text = file_content.decode("utf-8")
              except UnicodeDecodeError:
                  print_if_verbose(
                      self._verbose, f"could not decode {file_path} as utf-8"
                  )
                  continue

              print_if_verbose(
                  self._verbose,
                  f"got {len(decoded_text)} characters"
                  + f"- adding to documents - {file_path}",
              )
              document = Document(
                  text=decoded_text,
                  doc_id=None,
                  extra_info={
                      "file_path": rel_path,
                      "file_name": os.path.basename(file_path),
                  },
              )
              documents.append(document)

      return documents

    def _parse_supported_file(
        self, file_path: str, file_content: bytes, tree_sha: str, tree_path: str
    ) -> Optional[Document]:
        """
        Parse a file if it is supported by a parser.

        :param `file_path`: path of the file in the repo
        :param `file_content`: content of the file
        :return: Document if the file is supported by a parser, None otherwise
        """
        file_extension = get_file_extension(file_path)
        parser = DEFAULT_FILE_EXTRACTOR.get(file_extension)
        if parser is not None:
            parser.init_parser()
            print_if_verbose(
                self._verbose,
                f"parsing {file_path}"
                + f"as {file_extension} with "
                + f"{parser.__class__.__name__}",
            )
            with tempfile.TemporaryDirectory() as tmpdirname:
                with tempfile.NamedTemporaryFile(
                    dir=tmpdirname,
                    suffix=f".{file_extension}",
                    mode="w+b",
                    delete=False,
                ) as tmpfile:
                    print_if_verbose(
                        self._verbose,
                        "created a temporary file"
                        + f"{tmpfile.name} for parsing {file_path}",
                    )
                    tmpfile.write(file_content)
                    tmpfile.flush()
                    tmpfile.close()
                    try:
                        parsed_file = parser.parse_file(pathlib.Path(tmpfile.name))
                        if isinstance(parsed_file, ImageParserOutput):
                            raise ValueError(
                                "Reader does not support ImageParserOutput"
                            )
                        parsed_file = "\n\n".join(parsed_file)
                    except Exception as e:
                        print_if_verbose(
                            self._verbose, f"error while parsing {file_path}"
                        )
                        logger.error(
                            "Error while parsing "
                            + f"{file_path} with "
                            + f"{parser.__class__.__name__}:\n{e}"
                        )
                        parsed_file = None
                    finally:
                        os.remove(tmpfile.name)
                    if parsed_file is None:
                        return None
                    return Document(
                        text=parsed_file,
                        doc_id=tree_sha,
                        extra_info={
                            "file_path": file_path,
                            "file_name": tree_path,
                        },
                    )
        return None
