(this["webpackJsonpoptimizers-sign-language"]=this["webpackJsonpoptimizers-sign-language"]||[]).push([[0],{448:function(e,t,a){e.exports=a(623)},453:function(e,t,a){},454:function(e,t,a){},586:function(e,t){},587:function(e,t){},595:function(e,t){},598:function(e,t){},599:function(e,t){},623:function(e,t,a){"use strict";a.r(t);var n=a(1),r=a.n(n),l=a(60),i=a.n(l),o=(a(453),a(631)),c=(a(454),a(7)),s=a(172),u=a(632),m=a(287),d=a(373),p=a.n(d),f=a(634),g=a(228),b=a(4),h=a.n(b),E=a(12),v=a(635),y=a(636),w=a(637),k=a(139),j=a(627),x=a(141),O=a(374),_=a.n(O);function S(e,t,a){return A.apply(this,arguments)}function A(){return(A=Object(E.a)(h.a.mark((function e(t,a,n){var r,l,i,o,c,s,u,m;return h.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=new Date,l=a.predict(n),i=new Date,o=i.getTime()-r.getTime(),c=l.gather(0),s=t.postprocess(c),u=s.probabilities,m=s.prediction,e.abrupt("return",{time:o,probabilities:u,prediction:m});case 7:case"end":return e.stop()}}),e)})))).apply(this,arguments)}var N=a(629),F=a(628),I=a(630),z=a(633),L=a(375),C=a.n(L),T=N.a.Column;function W(e){var t=e.probabilities,a=e.prediction,n=e.top_n||10,l=C()(t,["probability"],["desc"]).map((function(e){return Object(g.a)({key:e.label},e)})).slice(0,n);return r.a.createElement(N.a,{dataSource:l,className:"inference-results",pagination:!1},r.a.createElement(T,{title:"Label",dataIndex:"label",key:"label",render:function(e){return r.a.createElement(r.a.Fragment,null,e===a?r.a.createElement("b",null,e):r.a.createElement("span",null,e))}}),r.a.createElement(T,{title:"Probability",dataIndex:"probability",key:"probability",render:function(e){return r.a.createElement(j.a,null,r.a.createElement(F.a,{span:12},r.a.createElement(I.a,{min:0,max:1,step:.01,value:e.toFixed(3)})),r.a.createElement(F.a,{span:4},r.a.createElement(z.a,{min:0,max:1,step:.01,value:e.toFixed(3),style:{width:"68px"}})))}}))}function U(e){var t={time:-1,probabilities:[],prediction:null,loading:!1},a=Object(n.useState)(t),l=Object(c.a)(a,2),i=l[0],o=l[1],u=e.model.imgSize,m=Object(n.useRef)(null),d=Object(n.useState)(!0),p=Object(c.a)(d,2),b=p[0],O=p[1];function A(){return(A=Object(E.a)(h.a.mark((function t(a){var n,r;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,_()(e.picture.base64data,{maxWidth:e.model.imgSize,crop:!0,canvas:!0,cover:!0});case 2:if(n=t.sent,m.current){t.next=5;break}return t.abrupt("return",console.warn("No canvas (drawimg)"));case 5:r=m.current.getContext("2d"),a?(console.log("crop!"),r.drawImage(n.image,-16,-16,256,256)):r.drawImage(n.image,0,0);case 7:case"end":return t.stop()}}),t)})))).apply(this,arguments)}function N(){return(N=Object(E.a)(h.a.mark((function a(){var n,r,l,i;return h.a.wrap((function(a){for(;;)switch(a.prev=a.next){case 0:if(o(Object(g.a)({},t,{loading:!0})),n=e.session,r=e.model,m.current){a.next=4;break}return a.abrupt("return",console.warn("No canvas (inferimg)"));case 4:return l=r.tensor(m.current),a.next=7,S(r,n,l);case 7:i=a.sent,console.log("inference result",i),setTimeout((function(){o(Object(g.a)({},i,{loading:!1}))}),750);case 10:case"end":return a.stop()}}),a)})))).apply(this,arguments)}Object(n.useEffect)((function(){e.picture.base64data&&function(e){A.apply(this,arguments)}(e.crop)}),[e.picture.base64data,e.model.imgSize,e.session]);var F=function(){return r.a.createElement(k.a,{title:"Remove picture"},r.a.createElement(s.a,{onClick:function(){return e.onRemove()},type:"text",icon:r.a.createElement(v.a,null)}))},I=i.loading,z=i.time,L=i.probabilities,C=i.prediction,T=function(){var t=!e.session||!e.picture.base64data,a="Perform inference";return e.session||(a="No model session available"),e.picture.base64data||(a="No image loaded"),r.a.createElement(r.a.Fragment,null,r.a.createElement(j.a,null,r.a.createElement(k.a,{title:a},r.a.createElement(s.a,{onClick:function(){return function(){return N.apply(this,arguments)}()},loading:I,disabled:t},"Inference"))),r.a.createElement(j.a,null,r.a.createElement("small",{style:{color:"#ccc"}},-1!==z?"Inference took ".concat(z,"ms"):r.a.createElement(r.a.Fragment,null,"\xa0"))))},U=function(){return b?r.a.createElement(s.a,{onClick:function(){return O(!1)},type:"text",icon:r.a.createElement(y.a,null)}):r.a.createElement(s.a,{onClick:function(){return O(!0)},type:"text",icon:r.a.createElement(w.a,null)})};return r.a.createElement(f.b.Item,{actions:[r.a.createElement(F,null),r.a.createElement(T,null)],className:"App-picitem"},r.a.createElement(f.b.Item.Meta,{title:e.picture.file.name.replace("_","-"),description:"".concat(u," x ").concat(u),avatar:e.picture.base64data?r.a.createElement("canvas",{ref:m,width:u,height:u,style:{minWidth:50,maxWidth:140}}):r.a.createElement(x.a,{description:"Image could not be loaded",style:{margin:"20px"}})}),r.a.createElement("div",{className:"ant-list-item-collapse"},r.a.createElement(U,null)),r.a.createElement(W,{probabilities:L,prediction:C,top_n:b?3:10}))}var R=function(e){var t=Object(n.useState)([]),a=Object(c.a)(t,2),l=a[0],i=a[1],o=Object(n.useRef)(null),s=function(e){return fetch(e).then((function(e){return e.blob()})).then((function(t){return new Promise((function(a,n){var r=t.type,l=e.split("/").pop(),i=new File([t],l,{type:r});r.startsWith("image")||(console.warn("Could not load picture `".concat(i.name,"` ")+"from url `".concat(e,"`.")),a({file:i,base64data:null}));var o=new FileReader;o.onloadend=function(){return a({file:i,base64data:o.result})},o.onerror=n,o.readAsDataURL(t)}))}))};return Object(n.useEffect)((function(){e.pictureUrls&&Promise.all(e.pictureUrls.map(s)).then(i)}),[e.pictureUrls]),r.a.createElement("div",null,r.a.createElement(f.b,{className:"App-piclist",dataSource:l,renderItem:function(t){return r.a.createElement(U,{picture:t,onRemove:function(){return function(e){o.current.removeImage(e.base64)}(t)},session:e.session,model:e.model,crop:e.crop})}}),r.a.createElement("div",{className:"App-imgupload",style:{display:e.pictureUrls?"none":"inline"}},r.a.createElement(p.a,{onChange:function(e,t){var a=e.map((function(e,a){return{file:e,base64data:t[a]}}));i(a)},ref:o})))},M=a(171);var P=function(e){var t=Object(n.useState)({msg:"No model",loading:!1,success:!1,session:null,feedback:"Load the model to start making inferences."}),a=Object(c.a)(t,2),l=a[0],i=a[1];Object(n.useEffect)((function(){l.loading&&M.d(e.modelFile).then((function(e){console.log("Model successfully loaded."),setTimeout((function(){i({msg:"Model successfully loaded",feedback:"TensorFlow.js is ready for live inferences.",success:!0,session:e})}),750)}),(function(e){i({msg:"Oops, model could not be loaded",feedback:e.message,loading:!1,failure:!0}),console.warn("Model failed to load",e)}))}),[e.modelFile,l.loading]);var o=e.modelFile,d=o&&o.replace(/^.*[\\/]/,"");return r.a.createElement("div",{style:{background:"white",margin:"50px 0"}},r.a.createElement("div",{style:{textAlign:"center"}},r.a.createElement("div",{style:{margin:"10px",display:"inline"}},d),r.a.createElement(s.a,{onClick:function(){return i({msg:"Loading...",loading:!0,success:!0})},disabled:l.loading},"Load model"),r.a.createElement(u.a,{status:l.success?"success":l.failure?"error":"info",title:l.msg,subTitle:r.a.createElement("code",null,l.feedback),icon:l.loading&&r.a.createElement(m.a,{style:{height:72}})})),e.children&&(e.children.map?e.children:[e.children]).map((function(t,a){return t.type===R?r.a.cloneElement(t,{key:a,session:l.session,model:e.model,crop:e.crop}):t})))},B=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","del","nothing","space"],D={imgSize:64,tensor:function(e){var t=M.a.fromPixels(e),a=M.c.resizeBilinear(t,[64,64]),n=M.b(a,"float32"),r=M.e(Array.from(n.dataSync()),[1,64,64,3]),l=r.mean().dataSync();return(r=r.sub(l)).sub(l).pow(2).div(r.size).sqrt(),r},postprocess:function(e){var t=e.dataSync();console.log("probs",t);var a=e.argMax().dataSync()[0];console.log("prediction",B[a]);for(var n=[],r=0;r<t.length;r++)n.push({probability:t[r],label:B[r]});return{probabilities:n,prediction:a}}},G=o.a.Text,H=o.a.Paragraph,J=o.a.Link;var K=function(){var e="/optimizers-sign-language";return r.a.createElement("article",{className:"App"},r.a.createElement("header",{className:"App-header"},r.a.createElement("h1",null,"Benchmarking Optimizers for Sign Language detection"),r.a.createElement("h4",null,r.a.createElement(G,{type:"secondary"},"Using Deep Learning with Keras"))),r.a.createElement(H,null,"Hey! Welcome to a live demonstration page of how our trained network performs. We trained a network on the ASL sign language ",r.a.createElement(J,{href:"https://kaggle.com/grassknoted/asl-alphabet"},"dataset"),", aiming to differentiate between 29 classes. We demonstrate the network trained using the ",r.a.createElement("b",null,"Adam")," optimizer, which yielded reasonable validation classification performance; about 90% accuracy. A learning rate of 0.001 was used, all other hyperparameters were standard. Let's see how well it performs, in an interactive way. "),r.a.createElement(P,{modelFile:e+"/adam/AdamOptimizer-NN.json",model:D},r.a.createElement(H,null,"Let's first test the model on images it has seen before, training images. It should be able to get these predicted correctly relatively easily."),r.a.createElement(R,{pictureUrls:[e+"/asl_alphabet_test/A_1.jpg",e+"/asl_alphabet_test/B_1.jpg",e+"/asl_alphabet_test/C_1.jpg"]}),r.a.createElement(H,null,"Next, we can predict images using ",r.a.createElement("b",null,"unseen")," data, test data. "),r.a.createElement(R,{pictureUrls:[e+"/asl_alphabet_test/E_test.jpg",e+"/asl_alphabet_test/F_test.jpg",e+"/asl_alphabet_test/G_test.jpg",e+"/asl_alphabet_test/H_test.jpg"]}),r.a.createElement(H,null,"Or optionally: upload your own images to predict! Try to make a clear photo and see whether our network is able to predict correctly."),r.a.createElement(R,null)),r.a.createElement(H,null,"This project uses ",r.a.createElement(J,{href:"https://www.tensorflow.org/js"},"TensorFlow.js")," to make live inferences in the browser. Our trained Keras model was converted using the ",r.a.createElement(G,{code:!0},"tfjs-converter"),", and then loaded up into this React.js application."),r.a.createElement(H,null,"Project built as part of the Deep Learning course ",r.a.createElement(G,{code:!0},"WMCS001-05")," taught at the University of Groningen. "),r.a.createElement(H,null,r.a.createElement("small",null,r.a.createElement(G,{type:"secondary"},"> All our code is available on\xa0",r.a.createElement(J,{href:"https://github.com/dunnkers/optimizers-sign-language"},"Github ",r.a.createElement("img",{src:e+"/github32.png",alt:"Github logo",style:{width:16,verticalAlign:"text-bottom"}}))))))};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));i.a.render(r.a.createElement(r.a.StrictMode,null,r.a.createElement(K,null)),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))}},[[448,1,2]]]);
//# sourceMappingURL=main.6a329ea0.chunk.js.map