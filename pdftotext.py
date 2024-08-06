from structml import line_heal
import re
from rich.progress import track
from rich.progress import Progress

from concurrent.futures import ThreadPoolExecutor

import pypdfium2 as pdfium
from pdftext.extraction import plain_text_output


# Use pdfium to return the text from each page of the PDF as a list of strings
def split_pdf_into_pages(pdf_path, verbose=False):
    output_text = []

    parsed_pdf = pdfium.PdfDocument(pdf_path)

    # for page_number in track(range(len(parsed_pdf)), description="Extracting text from PDF pages") if verbose else range(len(parsed_pdf)):

    #     page_text = plain_text_output(parsed_pdf, sort=False, hyphens=True, page_range=[page_number])

    #     output_text.append({"page_number": page_number, "text": page_text.split("\n")})

    if verbose:
        with Progress() as progress:
            task = progress.add_task("[green]Extracting text from PDF pages...", total=len(parsed_pdf))
            with ThreadPoolExecutor() as executor:
                for page_number in range(len(parsed_pdf)):
                    page_text = plain_text_output(parsed_pdf, sort=False, hyphens=True, page_range=[page_number])
                    output_text.append({"page_number": page_number, "text": page_text.split("\n")})
                    progress.update(task, advance=1)
    else:
        with ThreadPoolExecutor() as executor:
            for page_number in range(len(parsed_pdf)):
                page_text = plain_text_output(parsed_pdf, sort=False, hyphens=True, page_range=[page_number])
                output_text.append({"page_number": page_number, "text": page_text.split("\n")})

    return output_text

def clean_pages(pages, verbose=False):
    # Sort the text by page number
    pages.sort(key=lambda x: x["page_number"])

    # Remove any pages that contain nothing
    pages = [page for page in pages if page["text"]]

    # Remove any pages that contain only empty lines
    pages = [page for page in pages if any([line for line in page["text"] if line])]

    # Compare the first 3 lines of each page with the first 8 lines of every other page
    for i in track(range(len(pages))) if verbose else range(len(pages)):
        for j in range(len(pages)):
            if i == j:
                continue
            for line in pages[i]["text"][:3]:
                matching_characters = 0
                for other_line in pages[j]["text"][:8]:
                    # Calculate the number of characters that match
                    matching_characters = sum([1 for char1, char2 in zip(line, other_line) if char1 == char2])
                    # If more than 50% of the characters match, then assume that the line is a header and remove it and all the matching lines from the other pages
                    try:
                        if matching_characters / len(line) > 0.5:
                            # Replace line with an empty string
                            pages[i]["text"] = [l for l in pages[i]["text"] if l != line]
                    except ZeroDivisionError:
                        pass
                try:
                    if matching_characters / len(line) > 0.5:
                        pages[i]["text"] = [l for l in pages[i]["text"] if l != line]
                except ZeroDivisionError:
                    pass

                # If the line is just numbers, dashes or a combination of both, then remove it
                if re.match(r"^\d+-*$", line):
                    pages[i]["text"] = [l for l in pages[i]["text"] if l != line]

    # Do the same as above, but for the last 3 lines of each page
    for i in track(range(len(pages))) if verbose else range(len(pages)):
        for j in range(len(pages)):
            if i == j:
                continue
            for line in pages[i]["text"][-3:]:
                matching_characters = 0
                for other_line in pages[j]["text"][-8:]:
                    matching_characters = sum([1 for char1, char2 in zip(line, other_line) if char1 == char2])
                    try:
                        if matching_characters / len(line) > 0.5:
                            pages[i]["text"] = [l for l in pages[i]["text"] if l != line]
                    except ZeroDivisionError:
                        pass
                try:
                    if matching_characters / len(line) > 0.5:
                        pages[i]["text"] = [l for l in pages[i]["text"] if l != line]
                except ZeroDivisionError:
                    pass

                # If the line is just numbers, dashes or a combination of both, then remove it
                if re.match(r"^\d+-*$", line):
                    pages[i]["text"] = [l for l in pages[i]["text"] if l != line]

    return pages

def pdf_to_text(pdf_path, clean=True, slow_reformat=False, verbose=False):
    output_text = split_pdf_into_pages(pdf_path, verbose=verbose)

    output_text = clean_pages(output_text, verbose=False)

    output_text = "\n\n\n".join(["\n".join(page["text"]) for page in output_text])

    if slow_reformat:
        output_text = line_heal.parse(output_text, verbose=verbose)


    return output_text