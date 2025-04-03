import pytest
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import xml.etree.ElementTree as ET
import os


# Run pytest programmatically and collect results
def run_tests():
    # Run pytest and generate JUnit XML
    pytest.main(["test_app.py", "--junitxml=test_results.xml"])

    passed = 0
    failed = 0
    skipped = 0

    # Parse the JUnit XML file to count passed/failed tests
    if os.path.exists("test_results.xml"):
        try:
            tree = ET.parse("test_results.xml")
            root = tree.getroot()
            for testcase in root.findall(".//testcase"):
                # Check if the test passed (no failure or error)
                failure = testcase.find("failure")
                error = testcase.find("error")
                skipped_elem = testcase.find("skipped")

                if skipped_elem is not None:
                    skipped += 1
                elif failure is not None or error is not None:
                    failed += 1
                else:
                    passed += 1
        except Exception as e:
            print(f"Error parsing test_results.xml: {e}")

    return passed, failed, skipped


# Generate a pie chart of test results
def generate_chart(passed, failed, skipped):
    # Check if there are any tests to display
    total = passed + failed + skipped
    if total == 0:
        return None  # No chart if no tests were run

    labels = ['Passed', 'Failed', 'Skipped']
    sizes = [passed, failed, skipped]
    colors = ['#00FF00', '#FF0000', '#FFFF00']  # Green for passed, red for failed, yellow for skipped
    explode = (0.1, 0, 0)  # Explode the "Passed" slice

    # Filter out zero values to avoid empty pie chart issues
    filtered_labels = [label for label, size in zip(labels, sizes) if size > 0]
    filtered_sizes = [size for size in sizes if size > 0]
    filtered_colors = [color for color, size in zip(colors, sizes) if size > 0]
    filtered_explode = [exp for exp, size in zip(explode, sizes) if size > 0]

    if not filtered_sizes:
        return None  # No chart if all sizes are zero

    plt.figure(figsize=(6, 6))
    plt.pie(filtered_sizes, explode=filtered_explode, labels=filtered_labels, colors=filtered_colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Test Case Results")

    # Save the chart to a BytesIO object and encode it as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic


def generate_chart(passed, failed, skipped):
    total = passed + failed + skipped
    if total == 0:
        return None  # No chart if no tests were run

    labels = ['Passed', 'Failed', 'Skipped']
    counts = [passed, failed, skipped]
    colors = ['#00FF00', '#FF0000', '#FFFF00']  # Green for passed, red for failed, yellow for skipped

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=colors)
    plt.title("Test Case Results")
    plt.ylabel("Number of Test Cases")

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Save the chart to a BytesIO object and encode it as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic

# Embed the chart into the HTML report
def embed_chart_in_report(graphic):
    with open("report.html", "r") as f:
        content = f.read()

    # Add custom CSS for color-coding
    custom_css = """
    <style>
        .passed { background-color: #00FF00; color: black; }
        .failed { background-color: #FF0000; color: white; }
        .skipped { background-color: #FFFF00; color: black; }
    </style>
    """

    # Add the chart or a fallback message
    if graphic:
        chart_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{graphic}" alt="Test Results Chart"/></div><hr>'
    else:
        chart_html = '<div style="text-align: center; color: red;"><p>No test results available to display a chart.</p></div><hr>'

    new_content = content.replace('<head>', f'<head>{custom_css}').replace('<body>', f'<body>{chart_html}')

    with open("report.html", "w") as f:
        f.write(new_content)


if __name__ == "__main__":
    # Run tests and get results
    passed, failed, skipped = run_tests()
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    # Generate the chart
    graphic = generate_chart(passed, failed, skipped)

    # Embed the chart in the HTML report
    embed_chart_in_report(graphic)