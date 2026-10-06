#!/usr/bin/env python3
import argparse,json
from biojev.systemone_native.ollama import post_systemone,systemone_payload,validate_systemone_response
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--base-url",default="http://127.0.0.1:11434"); a=ap.parse_args()
    payload=systemone_payload(a.model,
      {"patient":"Fever, productive cough, and a new lobar infiltrate.","context":"Adult outpatient vignette."},
      {
        "diagnosis":{"type":"choice","instructions":"Which diagnosis is best supported?","criteria":{"pneumonia":"Community-acquired pneumonia","asthma":"Acute asthma exacerbation","migraine":"Migraine"}},
        "infection":{"type":"noul","instructions":"Is an infectious pulmonary process supported?","criteria":{"false":"No infectious pulmonary process is supported.","true":"An infectious pulmonary process is supported."}},
        "severity":{"type":"score","instructions":"How severe is the presentation?","criteria":["Low","Moderate","High"]},
      })
    result=post_systemone(payload,a.base_url)
    validate_systemone_response(result,{"diagnosis":"choice","infection":"noul","severity":"score"})
    print(json.dumps(result,indent=2,ensure_ascii=False)); print("\nSPRINT 8 OLLAMA SYSTEM ONE E2E: PASS")
if __name__=="__main__": main()
