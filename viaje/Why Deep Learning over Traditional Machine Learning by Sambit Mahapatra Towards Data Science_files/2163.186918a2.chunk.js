(self.webpackChunklite=self.webpackChunklite||[]).push([[2163],{75374:(e,t,n)=>{"use strict";n.d(t,{f:()=>Z,N:()=>X});var a=n(63038),i=n.n(a),r=n(67294),o=n(5977),l=n(38352),u=n(77355),c=n(8431),s=function(e){var t=e.title;return r.createElement(u.x,{width:"100%",padding:"0 20px"},r.createElement(u.x,{display:"flex",justifyContent:"space-between",marginBottom:"8px"},r.createElement(c.Lh,null,t)),r.createElement(l.oK,{paddingTopBottom:"0"}))},d=n(35473),m=n(14818),p=n(93310),f=n(87691),v=n(18627),y=n(14646),g=n(78249),h=n(87498),k=n(26679),b={display:"block",textAlign:"left",width:"252px",textOverflow:"ellipsis",overflow:"hidden",whiteSpace:"nowrap"},x={fill:"rgba(0, 0, 0, 0.54)",display:"block",marginLeft:"5px",marginRight:"6px"},I=function(e){var t=e.type,n=e.name,a=e.url,i=e.imageId,o=e.query,l=e.queryId,c=e.index,s=e.itemId,I=e.algoliaIndexName,E=e.isFocus,N=e.isLastItem,w=e.description,C=(0,v.Av)(),P=(0,y.I)(),O=(0,k.Ir)({queryId:l,indexName:I}),S=r.useRef(null),T=r.useCallback((function(e){C.event("search.predictiveResultClicked",{type:t.toLowerCase(),query:o,path:e.currentTarget.href,rank:0,total:0}),O(s,c)}),[o,t,O,s,c]),q=r.useMemo((function(){return"TAG"===t?r.createElement("div",{className:P(x)},r.createElement(g.Z,null)):"COLLECTION"===t?r.createElement(m.z,{miroId:i||"",freezeGifs:!1,alt:n||"Publication avatar",diameter:24}):"USER"===t?r.createElement(m.z,{miroId:i||h.gG,alt:n||"",diameter:24,freezeGifs:!1}):r.createElement(r.Fragment,null)}),[n,i]),D=r.useMemo((function(){return r.createElement(f.F,{color:"DARKER",scale:"M"},r.createElement("span",{className:P(b)},n))}),[n]);return r.useEffect((function(){E&&S.current&&S.current.focus()}),[E,S.current]),r.createElement(u.x,{display:"flex",tag:"li",paddingLeft:"20px",paddingRight:"12px",marginBottom:N?void 0:"12px"},r.createElement(p.r,{ref:S,tabIndex:-1,linkStyle:"SUBTLE",href:a,onClick:T},r.createElement(d.Y,{avatar:q,title:D,description:w&&r.createElement(u.x,{tag:"span",textAlign:"left"},w),clampDescription:1})))},E=n(60671),N=n(75221),w=n(50458),C=n(63459),P={width:"252px",textOverflow:"ellipsis",overflow:"hidden",whiteSpace:"nowrap"},O=function(e){var t=e.post,n=e.queryId,a=e.index,i=e.isEntityRouter,o=e.isFocus,l=(0,y.I)(),c=r.useRef(null),s=(0,w.jVf)({id:t.postId},i),d=(0,k.Ir)({queryId:n,indexName:N.uw.POST}),m=r.useCallback((function(){d(t.objectID,a)}),[d,a,t.objectID]);return r.useEffect((function(){o&&c.current&&c.current.focus()}),[o,c.current]),r.createElement(u.x,{display:"flex",tag:"li",paddingLeft:"20px",paddingRight:"12px"},r.createElement(u.x,{marginBottom:"16px"},r.createElement(p.r,{ref:c,onClick:m,tabIndex:-1,linkStyle:"SUBTLE",href:s},r.createElement(f.F,{scale:"S",color:"DARKER"},r.createElement("div",{className:l(P)},t.title)),r.createElement(f.F,{scale:"S"},r.createElement(C.E,{timestamp:t.latestPublishedAt})))))};function S(e){var t=e.data,n=e.refIndex,a=e.isLastPopover,i=e.query,o=t.collections;return r.createElement(r.Fragment,null,r.createElement(s,{title:"Publications"}),r.createElement(u.x,{marginTop:"12px",marginBottom:a?void 0:"15px"},o.hits.map((function(e,t){var a=e.collectionId,l=e.objectID,u=e.imageId,c=e.name;return r.createElement(I,{isFocus:n===t,key:a,algoliaIndexName:N.uw.COLLECTION,index:t,itemId:l,queryId:o.queryId,type:"COLLECTION",name:c,imageId:u,url:(0,w.RHb)(a),query:i,isLastItem:o.hits.length-1===t})}))))}function T(e){var t=e.data,n=e.authDomain,a=e.entityType,i=e.refIndex,o=e.isLastPopover,l=e.query,c=t.tags;return r.createElement(r.Fragment,null,r.createElement(s,{title:"Tags"}),r.createElement(u.x,{marginTop:"12px",marginBottom:o?void 0:"15px"},c.hits.map((function(e,t){var o=e.displayText,u=e.objectID,s=e.tagSlug;return r.createElement(I,{isFocus:i===t,key:s,type:"TAG",algoliaIndexName:N.uw.TAG,index:t,itemId:u,queryId:c.queryId,name:o,url:(0,w.HYG)(s,n,a),query:l,isLastItem:c.hits.length-1===t})}))))}function q(e){var t=e.data,n=e.authDomain,a=e.refIndex,i=e.isLastPopover,o=e.query,l=t.users;return r.createElement(r.Fragment,null,r.createElement(s,{title:"People"}),r.createElement(u.x,{marginTop:"12px",marginBottom:i?void 0:"15px"},l.hits.map((function(e,t){var i=e.imageId,u=e.objectID,c=e.name,s=e.userId,d=e.username,m=(0,w.Qyn)(d,n);return r.createElement(I,{isFocus:a===t,key:s,type:"USER",algoliaIndexName:N.uw.USER,index:t,queryId:l.queryId,itemId:u,imageId:i,name:c,url:m,query:o,isLastItem:l.hits.length-1===t})}))))}function D(e){var t=e.data,n=e.isCollection,a=e.refIndex,i=e.entityType,o=e.isLastPopover,l=e.isCatalog,c=t.posts,d="Profile";l?d="List":n&&(d="Publication");var m="From this ".concat(d);return r.createElement(r.Fragment,null,r.createElement(s,{title:m}),r.createElement(u.x,{marginTop:"12px",marginBottom:o?void 0:"15px"},c.hits.map((function(e,t){return r.createElement(O,{key:"story-item-".concat(t),post:e,index:t,queryId:c.queryId,isEntityRouter:i!==E.Cr.DEFAULT,isFocus:a===t})}))))}function F(e){var t=e.data,n=e.authDomain,a=e.query,i=e.entityType,o=e.focusIndex,c=e.isCollection,s=e.isCatalog;if(!a)return null;var d=t.collections,m=t.tags,p=t.users,f=t.posts,v=p.hits.length>0,y=d.hits.length>0,g=m.hits.length>0,h=f.hits.length>0;return v||y||g||h?r.createElement(u.x,{width:"316px"},r.createElement(l.mX,{padding:"30px"},h&&r.createElement(D,{data:t,isCollection:c,isCatalog:s,isLastPopover:!y&&!g&&!v,refIndex:o,query:a,entityType:i}),v&&r.createElement(q,{data:t,authDomain:n,refIndex:o,isLastPopover:!y&&!g,query:a}),y&&r.createElement(S,{data:t,refIndex:null!==o?o-p.hits.length:null,isLastPopover:!g,query:a}),g&&r.createElement(T,{data:t,authDomain:n,entityType:i,refIndex:null!==o?o-p.hits.length-d.hits.length:null,isLastPopover:!0,query:a}))):null}var j=n(25735),L=n(73917),_=n(31889),A=n(43487),R=n(21638),B=n(13241),U=n(27460),V=n(71341),Q=n(68894),M=n(42140),H=function(e){return function(t){return{display:"flex",border:"1px solid ".concat(t.baseColor.border.lighter),borderRadius:"20px",width:e?"100%":"inherit"}}},K=function(e){return{boxShadow:"0px 2px 10px 0px #00000026",border:"1px solid ".concat(e.baseColor.background.normal)}},G=function(e){return{":after":{boxShadow:"-1px -1px 1px -1px #00000026",border:"1px solid ".concat(e.baseColor.background.normal)}}},z=function(e){return{border:"none",outline:"none",fontFamily:R.k2,fontSize:"14px",lineHeight:"20px",marginRight:"20px",width:"100%",padding:"8px 0 11px",backgroundColor:"transparent",color:e.baseColor.text.normal,"::placeholder":{color:e.baseColor.text.lighter}}},Y="searchResults";function X(e){var t=e.searchPlaceholder,n=e.isPopoverVisible,a=e.onQueryChange,l=e.entityType,c=e.authDomain,s=e.children,d=e.onKeyDown,m=e.watchQueryUrl,p=e.fullWidth,f=e.collectionSlug,v=e.mainSearchCategory,g=(0,y.I)(),h=(0,_.F)(),b=(0,o.TH)(),x=(0,M.dD)(b.search).q,I=r.useRef(null),E=r.useState(m?x:null),N=i()(E,2),C=N[0],P=N[1],O=(0,A.v9)((function(e){return e.navigation.hostname})),S=(0,w.gxh)({category:v,collectionSlug:f}),T=(0,V.h)({queryParams:{q:null!=C?C:""}}),q=r.useCallback((function(e){P(e.target.value),a(e.target.value)}),[a]),D=r.useCallback((function(e){"Enter"===e.key&&C&&(0,k._)(C),d?d(e):"Enter"===e.key?O===c?T(S):window.location.assign("https://".concat(c).concat(S,"?q=").concat(C)):"ArrowDown"===e.key&&I.current&&I.current.blur()}),[I.current,C,l,c,O,S,T,d]);return r.useEffect((function(){x&&m&&(P(x),a(x))}),[x,a,m]),r.createElement("div",{className:g(H(!!p))},s,r.createElement(u.x,{tag:"span",padding:"7px 7px 6px 8px"},r.createElement(B.Z,{fill:h.baseColor.fill.darker})),r.createElement("input",{role:"combobox","aria-controls":Y,"aria-expanded":n?"true":"false","aria-label":"search",tabIndex:0,className:g(z),placeholder:t,ref:I,value:null!=C?C:"",onChange:q,onKeyDown:D}))}function Z(e){var t=e.algoliaIndexes,n=e.algoliaQueryContext,a=e.mainSearchCategory,o=e.publisher,l=e.catalogId,u=e.hitsPerPagePerIndex,c=(0,v.Av)(),s=(0,Q.O)(!1),d=i()(s,3),m=d[0],p=d[1],f=d[2],y=(0,A.v9)((function(e){return e.config.productName})),g=(0,A.v9)((function(e){return e.config.algolia})),h=(0,A.v9)((function(e){return e.config.authDomain})),k=(0,A.v9)((function(e){return e.client.routingEntity.type})),b=r.useState(null),x=i()(b,2),I=x[0],E=x[1],N=r.useState(""),w=i()(N,2),C=w[0],P=w[1],O="Collection"===(null==o?void 0:o.__typename),S="Collection"===(null==o?void 0:o.__typename)?o.slug:void 0,T=!!(0,j.VB)({name:"can_view_unfiltered_search",placeholder:!1}),q=!T,D=(0,U.xY)(n,C,t,{filterForHighQuality:q,publisher:o,catalogId:l,hitsPerPagePerIndex:u}),_=i()(D,2),R=_[0],B=_[1],V=r.useCallback((function(e){P(e),e||f(),E(null),m||(p(),c.event("search.predictiveOpened",{})),c.event("search.predictiveQueried",{query:e})}),[T,f,p,g,C]),M=r.useCallback((function(){return r.createElement(F,{productName:y,entityType:k,authDomain:h,query:C,data:R,focusIndex:I,isCollection:O,isCatalog:!!l})}),[y,k,h,C,R,I,O]);return r.useEffect((function(){if(m){var e=function(e){var t="ArrowUp"===e.key,n="ArrowDown"===e.key;if(t||n){e.preventDefault();var a=B-1;if(null===I&&n)E(0);else if(null!==I){var i=(I+(n?1:-1))%B;E(i<0?a:i)}}};return window.addEventListener("keydown",e),function(){window.removeEventListener("keydown",e)}}}),[I,m,B]),r.createElement(X,{authDomain:h,entityType:k,isPopoverVisible:B>0&&m,searchPlaceholder:"Search",onQueryChange:V,collectionSlug:S,mainSearchCategory:a},r.createElement(L.J,{ariaId:Y,isVisible:B>0&&m,hide:f,arrowPadding:{left:32},popoverRenderFn:M,setMaxHeight:!0,popoverRules:K,arrowRules:G,placement:"bottom-start"},r.createElement(r.Fragment,null)))}},19308:(e,t,n)=>{"use strict";n.d(t,{b3:()=>l,Iq:()=>u});var a=n(319),i=n.n(a),r=n(68216),o=n(98007),l={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"CollectionFollowButton_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}}]}}]},u={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"CollectionFollowButton_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"slug"}},{kind:"FragmentSpread",name:{kind:"Name",value:"collectionUrl_collection"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SusiClickable_collection"}}]}}].concat(i()(r.nf.definitions),i()(o.Os.definitions))}},27460:(e,t,n)=>{"use strict";n.d(t,{tp:()=>h,xY:()=>P});var a,i=n(59713),r=n.n(i),o=n(12297),l=n.n(o),u=n(94301),c=n.n(u),s=n(67294),d=n(27517),m=n(6443),p=n(86994),f=n(3184),v=n(43487);function y(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function g(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?y(Object(n),!0).forEach((function(t){r()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):y(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}!function(e){e[e.User=1]="User",e[e.Author=2]="Author"}(a||(a={}));var h={numericFilters:"peopleType!=".concat(a.Author)},k={collection:"collections",post:"posts",user:"users",tag:"tags"},b="/1/indexes/*/queries",x="user",I="collection",E="post",N="numericFilters";function w(e,t){return"".concat(e,"-").concat(t)}var C={collections:{hits:[]},posts:{hits:[]},users:{hits:[]},tags:{hits:[]}};function P(e,t,n){var a=arguments.length>3&&void 0!==arguments[3]?arguments[3]:{},i=s.useRef(""),r=a.filterForHighQuality,o=a.filters,u=a.publisher,y=a.catalogId,P=a.hitsPerPagePerIndex,O=(null==u?void 0:u.id)||null,S=(0,m.H)(),T=S.value,q=S.loading,D=(0,p.c)(),F=D.loading,j=D.isBot,L=q||F,_=(0,d.I0)(),A=(0,v.v9)((function(e){return e.config.algolia})),R=(0,v.v9)((function(e){var n;return null===(n=e.algolia.queries[t])||void 0===n?void 0:n.status})),B="https://".concat(A.appId).concat(A.host).concat(b),U=w(e,t),V=w(e,i.current),Q=(0,v.b$)((function(e){var t;return null===(t=e.algolia.queries[V])||void 0===t?void 0:t.data})),M=(0,v.b$)((function(e){var t;return null===(t=e.algolia.queries[U])||void 0===t?void 0:t.data})),H=M||Q||C;s.useEffect((function(){M&&(i.current=t)}),[t,M,i.current]);var K=s.useMemo((function(){return l()(Object.values(H).map((function(e){return e.hits.length})))}),[H]),G=["web","inline"],z=s.useMemo((function(){var e=n.map((function(e){var n=e===E&&y,a=g({query:t,hitsPerPage:P&&P[e]||3,numericFilters:"",clickAnalytics:!L&&!j,analyticsTags:n?G.concat(["list"]):G},o);return e===E&&(y?a.filters="_tags:'list_".concat(y,"'"):O&&(a.facetFilters="Collection"===(null==u?void 0:u.__typename)?"collectionId:".concat(O):"authorId:".concat(O))),"tag"===e?a[N]="postCount>=1":e===x&&r?a.filters="highQualityUser:true":e===I&&r?a.filters="isHighQualityCollection:true":e===E&&(a.attributesToRetrieve="title,postId,latestPublishedAt"),e===x&&(a[N]=h.numericFilters),{indexName:A.indexPrefix+e,params:Object.keys(a).map((function(e){return e+"="+a[e]})).join("&")}}));return JSON.stringify({requests:e})}),[A,L,j,t,T,r,o]),Y=s.useCallback((function(e){var t=e.results,a={users:{hits:[]},collections:{hits:[]},tags:{hits:[]},posts:{hits:[]}};return n.forEach((function(e,n){var i,r;a[k[e]]={hits:(null===(i=t[n])||void 0===i?void 0:i.hits)||[],queryId:null===(r=t[n])||void 0===r?void 0:r.queryID}})),a}),[n]);return s.useEffect((function(){if(t&&!L&&"loading"!==R&&"error"!==R&&"success"!==R){var e={"X-Algolia-API-Key":A.apiKeySearch,"X-Algolia-Application-Id":A.appId};T&&(e["X-Algolia-UserToken"]=T.id),_((0,f.Zl)(U)),c()(B,{method:"POST",headers:e,body:z}).then((function(e){return e.json()})).then((function(e){e.status>=400?_((0,f.TT)(U)):_((0,f.Yy)(U,Y(e)))})).catch((function(){_((0,f.TT)(U))}))}}),[B,A,z,T,L,t,R,Y,U]),[H,K]}},71341:(e,t,n)=>{"use strict";n.d(t,{h:()=>o});var a=n(67294),i=n(5977),r=n(66411);function o(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},t=(0,i.k6)(),n=(0,r.pK)(),o=a.useCallback((function(a){var i=new URLSearchParams;if(n&&i.set("source",n),e.queryParams)for(var r in e.queryParams)i.set(r,e.queryParams[r]);var o={pathname:a,search:i.toString()?"?".concat(i.toString()):"",hash:e.hash,state:e.state};e.replace?t.replace(o):t.push(o)}),[t,n,e.queryParams,e.hash,e.state]);return o}},26679:(e,t,n)=>{"use strict";n.d(t,{_:()=>P,KM:()=>C,DL:()=>O,Bm:()=>T,IB:()=>k,_u:()=>S,Ir:()=>w,VR:()=>N});var a=n(319),i=n.n(a),r=n(63038),o=n.n(r),l=n(59713),u=n.n(l),c=n(21919),s=n(67294),d=n(36405),m=n(75221),p=n(78285),f=n(28959),v=n(27460),y={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"mutation",name:{kind:"Name",value:"SearchClickEventMutation"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"queryId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"String"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"indexName"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"AlgoliaIndexName"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"objectIds"}},type:{kind:"NonNullType",type:{kind:"ListType",type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"String"}}}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"positions"}},type:{kind:"NonNullType",type:{kind:"ListType",type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"Int"}}}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"searchClickEvent"},arguments:[{kind:"Argument",name:{kind:"Name",value:"queryId"},value:{kind:"Variable",name:{kind:"Name",value:"queryId"}}},{kind:"Argument",name:{kind:"Name",value:"indexName"},value:{kind:"Variable",name:{kind:"Name",value:"indexName"}}},{kind:"Argument",name:{kind:"Name",value:"objectIds"},value:{kind:"Variable",name:{kind:"Name",value:"objectIds"}}},{kind:"Argument",name:{kind:"Name",value:"positions"},value:{kind:"Variable",name:{kind:"Name",value:"positions"}}}]}]}}]};function g(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function h(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?g(Object(n),!0).forEach((function(t){u()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):g(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var k={users:"search_user",posts:"search_post",tags:"search_tag",publications:"search_publication",none:"search_post"},b={filters:"highQualityUser:true OR writtenByHighQulityUser:true"},x={filters:"writtenByHighQualityUser:true"},I=new f.Z("recent_searches"),E="queries";function N(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},t=e.isRecommendations,n=!!(0,d.P5)("can_view_unfiltered_search"),a=!n,i=s.useMemo((function(){var e=["web"];return t&&e.push("recommendations"),{clickAnalytics:!0,analyticsTags:e}}),[t]),r=s.useMemo((function(){return h(h({},a?b:{}),v.tp)}),[a]),o=s.useMemo((function(){return a?x:{}}),[a]);return s.useMemo((function(){var e;return e={},u()(e,m.uw.COLLECTION,i),u()(e,m.uw.POST,h(h({},o),i)),u()(e,m.uw.USER,h(h({},r),i)),u()(e,m.uw.TAG,h({numericFilters:"postCount>=1"},i)),u()(e,m.uw.BOOK_EDITION,{}),e}),[i,o,r])}function w(e){var t=e.queryId,n=e.indexName,a=(0,c.D)(y),i=o()(a,1)[0];return s.useCallback((function(e,a){t&&e&&i({variables:{queryId:t,indexName:n,positions:[a+1],objectIds:[e]}})}),[i,t,n])}function C(){return I.get(E)||[]}function P(e){var t=C();if(e){var n=[e].concat(i()(t.filter((function(t){return t!==e}))));I.set(E,n.slice(0,10))}}function O(e){var t=C();I.set(E,t.filter((function(t){return e!==t})))}function S(e,t,n){var a=new URLSearchParams(t.location.search);e?a.set("q",e):a.delete("q");var i=a.toString();n((0,p.kO)(i)),t.push({search:i})}function T(e){var t=e.split("/"),n=t.indexOf("search"),a=n>-1?t[n+1]:null;return a?["users","tags","posts","publications"].includes(a)?a:null:"none"}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/2163.186918a2.chunk.js.map