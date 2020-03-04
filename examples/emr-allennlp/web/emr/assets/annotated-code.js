// Tutorial scrolling UX
const annotatedCode = document.getElementById("annotated-code");

// Proceed with code annotation logic if container element exists:
if (annotatedCode) {
  const annotationContainer = document.getElementById("annotated-code__annotations");
  // Dynamically create top/bottom fade effects for annotation scrolling:
  let topFadeNode = document.createElement("li");
  let bottomFadeNode = document.createElement("li");
  topFadeNode.setAttribute("id", "annotated-code__top-fade");
  bottomFadeNode.setAttribute("id", "annotated-code__bottom-fade");
  annotationContainer.insertBefore(topFadeNode, annotationContainer.childNodes[0]);
  annotationContainer.appendChild(bottomFadeNode);

  // Set common element constants:
  const topFade = document.getElementById("annotated-code__top-fade");
  const bottomFade = document.getElementById("annotated-code__bottom-fade");
  const codeBlocks = document.querySelectorAll(".annotated-code__code-block");
  const annotations = document.querySelectorAll("#annotated-code__annotations li.annotation");

  // Set default distance from top of screen for determining which code block is auto-focused during scroll:
  let scrollTop = 0;
  let containerTopOffset = 0;
  let scrollOffset = 0;
  let focusThreshold = 300;

  function focusBlock(id) {
    const focusedCodeBlock = document.getElementById(`c${id}`);
    const focusedAnnotation = document.getElementById(`a${id}`);

    // Remove focus class on any code block or annotation element that has it:
    function resetFocus(array) {
      for (let i = 0; i < array.length; i++) {
        if (array[i].classList.contains("focused")) {
          array[i].classList.remove("focused");
        }
      }
    }
    resetFocus(codeBlocks);
    resetFocus(annotations);

    // Add focused class to the focused code/annotation pair:
    focusedCodeBlock.classList.add("focused");
    focusedAnnotation.classList.add("focused");

    // Set container offsets:
    const codeOffset = focusedCodeBlock.offsetTop;
    const annotationOffset = focusedAnnotation.offsetTop;
    const offset = annotationOffset - codeOffset;

    // Move annotation list to align focused annotation with focused code block:
    annotationContainer.style.transform = `translateY(-${offset}px)`;
    // Compensate for transform offset on sticky bottom/top fade-out elements:
    topFade.style.transform = `translateY(${offset}px)`;
    bottomFade.style.transform = `translateY(${offset}px)`;
  }

  // Update scroll-based offsets:
  function updateOffsets() {
    scrollTop = (window.pageYOffset !== undefined) ? window.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop;
    containerTopOffset = annotatedCode.offsetTop;
    scrollOffset = scrollTop - containerTopOffset;
  }

  // onScroll Event:
  window.addEventListener("scroll", function() {
    updateOffsets();
    const containerBottomOffset = containerTopOffset + annotatedCode.offsetHeight;
    // Make topFade "sticky"
    if (scrollTop >= containerTopOffset) {
      topFade.style.top = `${scrollOffset - 120}px`;
    } else {
      topFade.style.top = "";
    }
    // Make bottomFade "sticky
    if (scrollTop <= containerBottomOffset) {
      bottomFade.style.top = `${scrollOffset + window.innerHeight}px`;
    }
    if (scrollTop >= (containerBottomOffset - window.innerHeight)) {
      bottomFade.style.top = `${annotatedCode.offsetHeight}px`;
    }
    // Focus code block at appropriate scroll offset:
    for (let i = 0; i < codeBlocks.length; i++) {
      const thisId = codeBlocks[i].id;
      const thisCodeBlock = document.getElementById(thisId);
      const thisOffset = thisCodeBlock.offsetTop;

      if ((scrollOffset > (thisOffset - focusThreshold)) && (scrollOffset < (thisOffset + thisCodeBlock.offsetHeight - focusThreshold))) {
        focusBlock(thisId.replace("c",""));
        break;
      }
    }
  });

  // Iterate through element list and add mouse events that call focusBlock
  function buildEvents(array) {
    for (let i = 0; i < array.length; i++) {
      const thisId = array[i].id;
      array[i].addEventListener("mousemove", function() {
        updateOffsets();
        focusThreshold = document.getElementById(`c${thisId.replace(/c|a/g,"")}`).offsetTop - scrollOffset;
        focusBlock(thisId.replace(/c|a/g,""));
      });
    }
  }
  buildEvents(codeBlocks);
  buildEvents(annotations);

  // Focus first code block/annotation pair by default
  focusBlock(document.querySelector(".annotated-code__code-block:first-child").id.replace(/c|a/g,""));
}