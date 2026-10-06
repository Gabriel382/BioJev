
from biojev.systemone_native.ollama import validate_systemone_response

def test_shapes():
    validate_systemone_response(
      {"answers":{
        "a":{"type":"choice","choice":"x","probabilities":{"x":.7,"y":.3},"confidence":.1},
        "b":{"type":"noul","noul":.9},
        "c":{"type":"score","score":1.2,"probabilities":{"0":.1,"1":.6,"2":.3},"confidence":.2},
      }},
      {"a":"choice","b":"noul","c":"score"})
