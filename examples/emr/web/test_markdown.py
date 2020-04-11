import markdown 


def test():
    text = 'Hello world github/linguist#1 **cool**, and #1!'
    mode = 'gfm'
    context = 'github/gollum'
    html = markdown.to_html(text, mode, context)
    assert html == '<p>Hello world <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1012654" data-permission-text="Title is private" data-url="https://github.com/github/linguist/issues/1" data-hovercard-type="issue" data-hovercard-url="/github/linguist/issues/1/hovercard" href="https://github.com/github/linguist/issues/1">github/linguist#1</a> <strong>cool</strong>, and <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="183433" data-permission-text="Title is private" data-url="https://github.com/gollum/gollum/issues/1" data-hovercard-type="issue" data-hovercard-url="/gollum/gollum/issues/1/hovercard" href="https://github.com/gollum/gollum/issues/1">gollum#1</a>!</p>'
