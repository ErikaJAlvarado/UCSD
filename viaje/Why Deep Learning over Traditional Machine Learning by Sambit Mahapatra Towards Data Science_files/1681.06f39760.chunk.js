(self.webpackChunklite=self.webpackChunklite||[]).push([[1681],{55459:(e,n,t)=>{"use strict";t.d(n,{Z:()=>l});var i=t(67294);function a(){return(a=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e}).apply(this,arguments)}var o=i.createElement("path",{d:"M21.27 20.06a9.04 9.04 0 0 0 2.75-6.68C24.02 8.21 19.67 4 14.1 4S4 8.21 4 13.38c0 5.18 4.53 9.39 10.1 9.39 1 0 2-.14 2.95-.41.28.25.6.49.92.7a7.46 7.46 0 0 0 4.19 1.3c.27 0 .5-.13.6-.35a.63.63 0 0 0-.05-.65 8.08 8.08 0 0 1-1.29-2.58 5.42 5.42 0 0 1-.15-.75zm-3.85 1.32l-.08-.28-.4.12a9.72 9.72 0 0 1-2.84.43c-4.96 0-9-3.71-9-8.27 0-4.55 4.04-8.26 9-8.26 4.95 0 8.77 3.71 8.77 8.27 0 2.25-.75 4.35-2.5 5.92l-.24.21v.32a5.59 5.59 0 0 0 .21 1.29c.19.7.49 1.4.89 2.08a6.43 6.43 0 0 1-2.67-1.06c-.34-.22-.88-.48-1.16-.74z"});const l=function(e){return i.createElement("svg",a({width:29,height:29},e),o)}},70762:(e,n,t)=>{"use strict";t.d(n,{Y:()=>r,r:()=>d});var i=t(319),a=t.n(i),o=t(98007),l={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"RegWall_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"lockedSource"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SusiClickable_post"}}]}}].concat(a()(o.qU.definitions))},d={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"Wall_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"content"},arguments:[{kind:"Argument",name:{kind:"Name",value:"postMeteringOptions"},value:{kind:"Variable",name:{kind:"Name",value:"postMeteringOptions"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"isLockedPreviewOnly"}}]}},{kind:"Field",name:{kind:"Name",value:"isLocked"}},{kind:"Field",name:{kind:"Name",value:"isMarkedPaywallOnly"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PayWall_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"RegWall_post"}}]}}].concat(a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"PayWall_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"imageId"}}]}}]}}]),a()(l.definitions))},r={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"Wall_meteringInfo"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"MeteringInfo"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"RegWall_meteringInfo"}}]}}].concat(a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"RegWall_meteringInfo"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"MeteringInfo"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"postIds"}}]}}]))}},62549:(e,n,t)=>{"use strict";t.d(n,{t:()=>a});var i=t(319),a={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"DraftStatus_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"pendingCollection"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"BoldCollectionName_collection"}}]}},{kind:"Field",name:{kind:"Name",value:"statusForCollection"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"Field",name:{kind:"Name",value:"isPublished"}}]}}].concat(t.n(i)()([{kind:"FragmentDefinition",name:{kind:"Name",value:"BoldCollectionName_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}}]}}]))}},95538:(e,n,t)=>{"use strict";t.d(n,{U:()=>m,m:()=>s});var i=t(319),a=t.n(i),o=t(42423),l=t(79987),d={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"HighlighSegmentContext_paragraph"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Paragraph"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"ParagraphRefsMapContext_paragraph"}}]}}].concat(a()(l.p.definitions))},r=([{kind:"FragmentDefinition",name:{kind:"Name",value:"ActiveSelectionContext_highlight"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Quote"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SelectionMenu_highlight"}}]}}].concat(a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"SelectionMenu_highlight"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Quote"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"userId"}},{kind:"Field",name:{kind:"Name",value:"user"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"name"}}]}}]}}])),{kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"ActiveSelectionContext_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SelectionMenu_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostNewNoteCard_post"}}]}}].concat(a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"SelectionMenu_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"isPublished"}},{kind:"Field",name:{kind:"Name",value:"isLocked"}},{kind:"Field",name:{kind:"Name",value:"latestPublishedVersion"}},{kind:"Field",name:{kind:"Name",value:"visibility"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"allowNotes"}}]}}]}}]),a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"PostNewNoteCard_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"latestPublishedVersion"}}]}}]))}),s={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"InteractivePostBody_postPreview"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"extendedPreviewContent"},arguments:[{kind:"Argument",name:{kind:"Name",value:"truncationConfig"},value:{kind:"ObjectValue",fields:[{kind:"ObjectField",name:{kind:"Name",value:"previewParagraphsWordCountThreshold"},value:{kind:"IntValue",value:"400"}},{kind:"ObjectField",name:{kind:"Name",value:"minimumWordLengthForTruncation"},value:{kind:"IntValue",value:"150"}},{kind:"ObjectField",name:{kind:"Name",value:"truncateAtEndOfSentence"},value:{kind:"BooleanValue",value:!0}},{kind:"ObjectField",name:{kind:"Name",value:"showFullImageCaptions"},value:{kind:"BooleanValue",value:!0}},{kind:"ObjectField",name:{kind:"Name",value:"shortformPreviewParagraphsWordCountThreshold"},value:{kind:"IntValue",value:"30"}},{kind:"ObjectField",name:{kind:"Name",value:"shortformMinimumWordLengthForTruncation"},value:{kind:"IntValue",value:"30"}}]}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"bodyModel"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostBody_bodyModel"}}]}},{kind:"Field",name:{kind:"Name",value:"isFullContent"}}]}}]}}].concat(a()(o.Pk.definitions))},m={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"InteractivePostBody_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"content"},arguments:[{kind:"Argument",name:{kind:"Name",value:"postMeteringOptions"},value:{kind:"Variable",name:{kind:"Name",value:"postMeteringOptions"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"isLockedPreviewOnly"}},{kind:"Field",name:{kind:"Name",value:"bodyModel"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostBody_bodyModel"}},{kind:"Field",name:{kind:"Name",value:"paragraphs"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"HighlighSegmentContext_paragraph"}},{kind:"FragmentSpread",name:{kind:"Name",value:"NormalizeHighlights_paragraph"}}]}}]}}]}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"allowNotes"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostBody_creator"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"ActiveSelectionContext_post"}}]}}].concat(a()(o.Pk.definitions),a()(d.definitions),a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"NormalizeHighlights_paragraph"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Paragraph"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"text"}}]}}]),a()(o.v.definitions),a()(r.definitions))}},7834:(e,n,t)=>{"use strict";t.d(n,{b:()=>D});var i=t(67154),a=t.n(i),o=t(63038),l=t.n(o),d=t(28655),r=t.n(d),s=t(92471),m=t(38460),c=t(67294),u=t(38882),k=t(75119),v=t(86249),p=t(10374),S=t(37070),N=t(83363),f=t(59713),g=t.n(f),h=t(39202);function F(){var e=r()(["\n  fragment NormalizeHighlights_paragraph on Paragraph {\n    name\n    text\n  }\n"]);return F=function(){return e},e}function y(){var e=r()(["\n  fragment NormalizeHighlights_highlight on Quote {\n    endOffset\n    startOffset\n    paragraphs {\n      name\n      text\n    }\n  }\n"]);return y=function(){return e},e}function b(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);n&&(i=i.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,i)}return t}function E(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?b(Object(t),!0).forEach((function(n){g()(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):b(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function P(e,n,t){if(!e||"number"!=typeof n||"number"!=typeof t)return null;var i=n,a=t;return e.substr(i,a-i)}var _=(0,s.Ps)(y()),O=((0,s.Ps)(F()),t(6443)),I=t(6402);function C(){var e=r()(["\n  query InteractivePostBodyQuery($postId: ID!, $showNotes: Boolean!) {\n    post(id: $postId) {\n      id\n      highlights {\n        id\n        ...ActiveSelectionContext_highlight\n        ...HighlighSegmentContext_highlight\n        ...NormalizeHighlights_highlight\n        ...PostBody_highlight\n      }\n      privateNotes @include(if: $showNotes) {\n        ...PostBody_privateNote\n      }\n    }\n  }\n  ","\n  ","\n  ","\n  ","\n  ","\n"]);return C=function(){return e},e}var T=(0,s.Ps)(C(),v.UW,N.g8,S.XV,S.w6,_),D=c.forwardRef((function(e,n){var t,i,o,d=e.isAuroraPostPageEnabled,r=e.post,s=e.inlinePost,f=e.postBodyInserts,g=e.richTextStyle,F=e.markedUpBodyModel,y=e.shouldHideHighlightAnnotations,b=(0,O.H)().value,_=s&&!r?F||(null==s||null===(t=s.extendedPreviewContent)||void 0===t?void 0:t.bodyModel):null==r||null===(i=r.content)||void 0===i?void 0:i.bodyModel,C={creator:null==r?void 0:r.creator,isAuroraPostPageEnabled:d,postBodyInserts:f,ref:n,richTextStyle:g||"FULL_PAGE",postId:(null==r?void 0:r.id)||""},D=r&&r.creator&&r.creator.allowNotes||!1,w=(0,m.t)(T,{ssr:!1,variables:{postId:(null==r?void 0:r.id)||"",showNotes:D},notifyOnNetworkStatusChange:!0}),L=l()(w,2),x=L[0],B=L[1],M=B.called,j=B.data,A=null==j?void 0:j.post;if(c.useEffect((function(){r&&!M&&x()}),[r]),!_)return I.k.error("Expected post to have content or preview content."),null;var R=(null==r||null===(o=r.content)||void 0===o?void 0:o.bodyModel)&&(null==r?void 0:r.content.bodyModel.paragraphs)||[],H=function(e,n){return e&&n?e.map((function(e){var t=e.endOffset,i=e.startOffset,a=e.paragraphs&&e.paragraphs[0],o=function(e,n){return e?n.find((function(n){var t=n.name;return e===t})):null}(a.name,n);if(!a||!o)return null;var l=P(a.text,i,t),d=P(o.text,i,t);if(!l||!d)return null;if(l===d)return e;var r=o.text?o.text.indexOf(l):null;if("number"!=typeof r||r<0)return null;var s=r+l.length;return E(E({},e),{},{endOffset:s,startOffset:r})})).filter(h.b):e}(A&&A.highlights||[],R),U=A&&A.privateNotes||null;return c.createElement(p.yb,null,c.createElement(k.KN,null,c.createElement(v.Ms,{interactivePost:r,highlights:H,disableSelection:!r},c.createElement(N.U7,{highlights:H,paragraphs:R,viewer:b},c.createElement(u.c.Provider,{value:!0},c.createElement(S.yO,a()({},C,{bodyModel:_,highlights:H,privateNotes:U,shouldHideHighlightAnnotations:y})))))))}))},42423:(e,n,t)=>{"use strict";t.d(n,{Pk:()=>m,v:()=>c});var i=t(319),a=t.n(i),o=t(69948),l=t(54975),d=t(27048),r={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostNotesDetails_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"imageId"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"username"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserAvatar_user"}}]}}].concat(a()(d.W.definitions))},s={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostNotesMarkers_highlight"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Quote"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"endOffset"}},{kind:"Field",name:{kind:"Name",value:"paragraphs"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}}]}},{kind:"Field",name:{kind:"Name",value:"startOffset"}},{kind:"Field",name:{kind:"Name",value:"userId"}},{kind:"Field",name:{kind:"Name",value:"user"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostNotesDetails_user"}}]}}]}}].concat(a()(r.definitions))},m={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostBody_bodyModel"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"RichText"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"sections"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"startIndex"}},{kind:"Field",name:{kind:"Name",value:"textLayout"}},{kind:"Field",name:{kind:"Name",value:"imageLayout"}},{kind:"Field",name:{kind:"Name",value:"backgroundImage"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"originalHeight"}},{kind:"Field",name:{kind:"Name",value:"originalWidth"}}]}},{kind:"Field",name:{kind:"Name",value:"videoLayout"}},{kind:"Field",name:{kind:"Name",value:"backgroundVideo"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"videoId"}},{kind:"Field",name:{kind:"Name",value:"originalHeight"}},{kind:"Field",name:{kind:"Name",value:"originalWidth"}},{kind:"Field",name:{kind:"Name",value:"previewImageId"}}]}}]}},{kind:"Field",name:{kind:"Name",value:"paragraphs"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostBodySection_paragraph"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"normalizedBodyModel_richText"}}]}}].concat(a()(o.Fm.definitions),a()(l.gd.definitions))},c=([{kind:"FragmentDefinition",name:{kind:"Name",value:"PostBody_highlight"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Quote"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"paragraphs"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"normalizedBodyModel_highlight"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostBodySection_highlight"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostNotesMarkers_highlight"}}]}}].concat(a()(l.Cn.definitions),a()(o.rz.definitions),a()(s.definitions)),{kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostBody_creator"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostNotesMarkers_creator"}}]}}].concat(a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"PostNotesMarkers_creator"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}}]}}]))});[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostBody_privateNote"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Note"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"normalizedBodyModel_privateNote"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostBodySection_privateNote"}}]}}].concat(a()(l.EH.definitions),a()(o.ik.definitions))},93403:(e,n,t)=>{"use strict";t.d(n,{z:()=>d});var i=t(319),a=t.n(i),o=t(98007),l=t(84130),d={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"BookmarkButton_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"visibility"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SusiClickable_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"AddToCatalogBookmarkButton_post"}}]}}].concat(a()(o.qU.definitions),a()(l.G.definitions))}},1444:(e,n,t)=>{"use strict";t.d(n,{o:()=>m});var i=t(67294),a=t(39959),o=t(6443),l=t(75221),d=t(97217),r=t(43487),s=t(50458),m=function(e){var n=e.post,t=e.susiEntry,m=e.buttonStyle,c=n.id,u=n.visibility,k=(0,r.v9)((function(e){return e.config.authDomain}));return(0,o.H)().loading||u===d.Wn.UNLISTED?null:i.createElement(a.o,{kind:l.ej.POST,target:n,buttonStyle:m,susiEntry:t,susiActionUrl:(0,s.XET)(k,c)})}},95204:(e,n,t)=>{"use strict";t.d(n,{R:()=>u});var i=t(319),a=t.n(i),o=t(93403),l=t(10654),d=t(29053),r=t(15855),s=t(57572),m={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostFooterSocialPopover_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"mediumUrl"}},{kind:"Field",name:{kind:"Name",value:"title"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SharePostButton_post"}}]}}].concat(a()(s.o.definitions))},c=t(51277),u={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostFooterActionsBar_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"visibility"}},{kind:"Field",name:{kind:"Name",value:"isPublished"}},{kind:"Field",name:{kind:"Name",value:"allowResponses"}},{kind:"Field",name:{kind:"Name",value:"postResponses"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"count"}}]}},{kind:"Field",name:{kind:"Name",value:"isLimitedState"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"Field",name:{kind:"Name",value:"collection"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"BookmarkButton_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"MultiVote_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"ManageSubmission_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SharePostButtons_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostFooterSocialPopover_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"OverflowMenuButtonWithNegativeSignal_post"}}]}}].concat(a()(o.z.definitions),a()(l.x.definitions),a()(d.E.definitions),a()(r.E.definitions),a()(m.definitions),a()(c.v.definitions))}},67927:(e,n,t)=>{"use strict";t.d(n,{F:()=>_});var i=t(28655),a=t.n(i),o=t(67154),l=t.n(o),d=t(92471),r=t(67294),s=t(91743),m=t(51277),c=t(50455),u=t(1444),k=t(74543),v=t(62676),p=t(20210),S=t(10654),N=t(17878),f=t(3428),g=t(21232),h=t(6443),F=t(75399),y=t(77355),b=t(30020),E=t(66411);function P(){var e=a()(["\n  fragment PostFooterActionsBar_post on Post {\n    id\n    visibility\n    isPublished\n    allowResponses\n    postResponses {\n      count\n    }\n    isLimitedState\n    creator {\n      id\n    }\n    collection {\n      id\n    }\n    ...BookmarkButton_post\n    ...MultiVote_post\n    ...ManageSubmission_post\n    ...SharePostButtons_post\n    ...PostFooterSocialPopover_post\n    ...OverflowMenuButtonWithNegativeSignal_post\n  }\n  ","\n  ","\n  ","\n  ","\n  ","\n  ","\n"]);return P=function(){return e},e}var _=function(e){var n,t=e.post,i=e.useSocialPopover,a=e.shouldHideClapsText,o=void 0!==a&&a,d=r.useContext(g.f).openSidebar,m=!!(0,h.H)().value,c=t.isPublished,v=t.creator,S="clap_footer",P={post:t,source:"post_actions_footer"},_=(null==t||null===(n=t.postResponses)||void 0===n?void 0:n.count)||null;return r.createElement(r.Fragment,null,r.createElement(y.x,{display:"flex",justifyContent:"space-between",print:{display:"none"}},r.createElement(E.cW,{source:{name:"post_actions_footer"}},r.createElement(y.x,{display:"flex",flexDirection:"row",alignItems:"center"},r.createElement(y.x,{maxWidth:"155px"},r.createElement(N.e,null,r.createElement(p.S,{post:t,buttonStyle:"SUBTLE_MARGIN",hasDialog:!0,shouldShowResponsiveLabelText:!0,shouldHideClapsText:o,susiEntry:S,buttonColor:"LIGHTER",countScale:"S"})),r.createElement(N.s,null,r.createElement(p.S,{post:t,buttonStyle:"SUBTLE_MARGIN",hasDialog:!0,shouldHideClapsText:o,susiEntry:S,buttonColor:"LIGHTER",countScale:"S"}))),r.createElement(E.cW,{source:{name:"follow_footer",susiEntry:"follow_footer"}},r.createElement(y.x,{display:"flex",marginLeft:"24px"},r.createElement(f.h,{svgSize:"S",trackingData:{postId:t.id},responsesCount:_,allowResponses:t.allowResponses,isLimitedState:t.isLimitedState,handleClick:d,iconStylesOverride:{marginTop:"0px"},countStylesOverride:{marginLeft:"4px",marginTop:"0px"}}))))),r.createElement(y.x,{display:"flex",alignItems:"center"},c&&r.createElement(r.Fragment,null,i?r.createElement(k.$,l()({},P,{shareIconStyle:"ICON_SUBTLE"})):r.createElement(F.U,P),r.createElement(y.x,{flexGrow:"0",margin:i?"0 20px":"0 4px 0 5px"},r.createElement(E.cW,{source:{name:"post_actions_footer"}},r.createElement(u.o,{post:t,susiEntry:"bookmark_footer",buttonStyle:"ICON_SUBTLE"})))),t&&v&&m&&r.createElement(b._,{tooltipText:"More",targetDistance:10},r.createElement(s.t,{post:t,iconStyle:"ICON_SUBTLE"})))))};(0,d.Ps)(P(),c.z,S.x,v.En,F.E,k.u,m.v)},74543:(e,n,t)=>{"use strict";t.d(n,{$:()=>u,u:()=>k});var i=t(28655),a=t.n(i),o=t(92471),l=t(67294),d=t(12287),r=t(85805),s=t(37597),m=t(38352);function c(){var e=a()(["\n  fragment PostFooterSocialPopover_post on Post {\n    id\n    mediumUrl\n    title\n    ...SharePostButton_post\n  }\n  ","\n"]);return c=function(){return e},e}var u=function(e){var n=e.post,t=e.source,i=e.shareIconStyle,a=n.mediumUrl,o=n.title,c=n.id;return l.createElement(r.A,{ariaId:"postFooterSocialMenu",source:{name:t},url:a,title:o,ariaLabel:"Share Post",postId:c,iconStyle:i},(function(e){return l.createElement(l.Fragment,null,a&&l.createElement(l.Fragment,null,l.createElement(m.Sl,null,l.createElement(d._,{url:a,onClick:e,reportData:{postId:n.id},source:t,copyStyle:"INLINE"})),l.createElement(m.oK,{paddingTopBottom:"5px"})),l.createElement(m.Sl,{paddingTopBottom:"5px"},l.createElement(s.f,{socialPlatform:"TWITTER",buttonStyle:"LINK_INLINE_SHORT_LABEL",post:n})),l.createElement(m.Sl,{paddingTopBottom:"5px"},l.createElement(s.f,{socialPlatform:"FACEBOOK",buttonStyle:"LINK_INLINE_SHORT_LABEL",post:n})),l.createElement(m.Sl,{paddingTopBottom:"5px"},l.createElement(s.f,{socialPlatform:"LINKEDIN",buttonStyle:"LINK_INLINE_SHORT_LABEL",post:n})))}))},k=(0,o.Ps)(c(),s.o)},3428:(e,n,t)=>{"use strict";t.d(n,{h:()=>g});var i=t(59713),a=t.n(i),o=t(67294),l=t(30020),d=t(87691),r=t(18627),s=t(66411),m=t(14646);function c(){return(c=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e}).apply(this,arguments)}var u=o.createElement("path",{d:"M18 16.8a7.14 7.14 0 0 0 2.24-5.32c0-4.12-3.53-7.48-8.05-7.48C7.67 4 4 7.36 4 11.48c0 4.13 3.67 7.48 8.2 7.48a8.9 8.9 0 0 0 2.38-.32c.23.2.48.39.75.56 1.06.69 2.2 1.04 3.4 1.04.22 0 .4-.11.48-.29a.5.5 0 0 0-.04-.52 6.4 6.4 0 0 1-1.16-2.65v.02zm-3.12 1.06l-.06-.22-.32.1a8 8 0 0 1-2.3.33c-4.03 0-7.3-2.96-7.3-6.59S8.17 4.9 12.2 4.9c4 0 7.1 2.96 7.1 6.6 0 1.8-.6 3.47-2.02 4.72l-.2.16v.26l.02.3a6.74 6.74 0 0 0 .88 2.4 5.27 5.27 0 0 1-2.17-.86c-.28-.17-.72-.38-.94-.59l.01-.02z"});const k=function(e){return o.createElement("svg",c({width:24,height:24,viewBox:"0 0 24 24"},e),u)};var v=t(55459);function p(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);n&&(i=i.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,i)}return t}function S(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?p(Object(t),!0).forEach((function(n){a()(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):p(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var N=function(e,n){return"LIGHTER"===n?e.baseColor.fill.darker:e.baseColor.fill.lighter},f=function(e,n){return function(t){return{cursor:n?"not-allowed":"pointer",border:0,padding:"4px 0",display:"flex",alignItems:"center",fill:"LIGHTER"===e?t.baseColor.fill.lighter:t.baseColor.fill.light,":hover":n?void 0:{fill:N(t,e),"& p":{color:N(t,e)}}}}},g=function(e){var n=e.allowResponses,t=e.responsesCount,i=e.handleClick,a=e.trackingData,c=e.isLimitedState,u=e.iconStylesOverride,p=e.countStylesOverride,N=e.svgSize,g=void 0===N?"L":N,h=e.responsesCountColor,F=void 0===h?"LIGHTER":h,y=e.disabledTooltipText,b=void 0===y?"":y,E=e.responsesCountScale,P=void 0===E?"M":E,_=(0,m.I)(),O=(0,r.Av)(),I=(0,s.pK)();if(!n)return null;var C={opacity:c?.4:1},T="S"===g?o.createElement(k,{"aria-label":"responses",className:_([u])}):o.createElement(v.Z,{"aria-label":"responses",className:_([u])});return o.createElement(l._,{tooltipText:c?b:"Respond",targetDistance:15},o.createElement("button",{onClick:c?void 0:function(e){null==i||i(e),O.event("responses.viewAllClicked",S(S({},a),{},{source:I}))},className:_(f(F,c)),"aria-label":"responses"},T,!!t&&o.createElement(d.F,{scale:P,color:F},o.createElement("span",{className:"pw-responses-count ".concat(_([p,C]))},t))))}},21232:(e,n,t)=>{"use strict";t.d(n,{f:()=>i});var i=t(67294).createContext({addContinueThisThreadSidebar:function(){return null},openSidebar:function(){return null},closeSidebar:function(){return null}})},33384:(e,n,t)=>{"use strict";t.d(n,{T:()=>g});var i=t(67154),a=t.n(i),o=t(59713),l=t.n(o),d=t(67294),r=t(25735),s=t(23500),m=t(6729),c=t(93310),u=t(77355),k=t(47230),v=t(30020),p=t(87691);function S(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);n&&(i=i.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,i)}return t}function N(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?S(Object(t),!0).forEach((function(n){l()(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):S(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var f=function(e){return{display:"inline-flex",alignItems:"center",":hover path":{fill:e.baseColor.fill.normal}}};function g(e){var n,t=e.buttonStyle,i=e.socialPlatform,o=e.baseOnClick,l=e.href,S=(0,r.VB)({name:"enable_in_context_sharing",placeholder:!1}),g=m.sC[i];if(!g)return null;switch(n=N(N({},{"aria-label":"Share on ".concat(i.toLowerCase())}),{},{onClick:function(){o();var e=Math.max((window.outerHeight||200)/2-560,100),n=(window.outerWidth||200)/2-250;S?window.open(l,"Social Share Window","resizable,scrollbars,status,top=".concat(e,",left=").concat(n,",height=").concat(650,",width=").concat(650)):"LINKEDIN"===i?window.open(l,"LIP","resizable,scrollbars,status,top=".concat(e,",left=").concat(n,",height=").concat(560,",width=").concat(500)):window.open(l)}}),t){case"LINK_DISABLED":return d.createElement(v._,{tooltipText:"This feature is temporarily disabled.",targetDistance:15},d.createElement(s.T,{buttonStyle:t,socialPlatform:i}));case"LINK":case"LINK_SUBTLE":return d.createElement(c.r,n,d.createElement(s.T,{buttonStyle:t,socialPlatform:i}));case"LINK_INLINE":return d.createElement(c.r,n,d.createElement(s.T,{buttonStyle:t,socialPlatform:i}),d.createElement(u.x,{display:"inline",marginLeft:"8px",marginTop:"2px"},d.createElement(p.F,{scale:"S",tag:"span"},"Share on ",g)));case"LINK_INLINE_SHORT_LABEL":return d.createElement(c.r,a()({},n,{rules:f}),d.createElement(s.T,{buttonStyle:t,socialPlatform:i}),d.createElement(u.x,{display:"inline",marginLeft:"8px"},"Share on ",g));case"BUTTON_BRANDED":return d.createElement(k.z,a()({},n,{buttonStyle:"SOCIAL",size:"REGULAR",width:"212px"}),d.createElement(u.x,{display:"flex",alignItems:"center",justifyContent:"center"},d.createElement(s.T,{buttonStyle:t,socialPlatform:i}),"Share with ".concat(g)));default:return null}}},85805:(e,n,t)=>{"use strict";t.d(n,{A:()=>g});var i=t(63038),a=t.n(i),o=t(67294),l=t(38352),d=t(73917),r=t(93310),s=t(30020),m=t(18627),c=t(66411),u=t(31889);function k(){return(k=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e}).apply(this,arguments)}var v=o.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M15.22 4.93a.42.42 0 0 1-.12.13h.01a.45.45 0 0 1-.29.08.52.52 0 0 1-.3-.13L12.5 3v7.07a.5.5 0 0 1-.5.5.5.5 0 0 1-.5-.5V3.02l-2 2a.45.45 0 0 1-.57.04h-.02a.4.4 0 0 1-.16-.3.4.4 0 0 1 .1-.32l2.8-2.8a.5.5 0 0 1 .7 0l2.8 2.8a.42.42 0 0 1 .07.5zm-.1.14zm.88 2h1.5a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-10a2 2 0 0 1 2-2H8a.5.5 0 0 1 .35.14c.1.1.15.22.15.35a.5.5 0 0 1-.15.35.5.5 0 0 1-.35.15H6.4c-.5 0-.9.4-.9.9v10.2a.9.9 0 0 0 .9.9h11.2c.5 0 .9-.4.9-.9V8.96c0-.5-.4-.9-.9-.9H16a.5.5 0 0 1 0-1z",fill:"#000"});const p=function(e){return o.createElement("svg",k({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),v)};var S=t(68894),N=t(6729),f=function(e){var n=e.children,t=e.source;return t?o.createElement(c.cW,{source:t},n):o.createElement(o.Fragment,null,n)},g=function(e){var n,t=e.url,i=e.title,k=e.source,v=e.ariaId,g=e.children,h=e.ariaLabel,F=e.tooltipText,y=void 0===F?"":F,b=e.postId,E=e.listId,P=e.iconStyle,_=void 0===P?"ICON":P,O=(0,u.F)(),I=(0,S.O)(!1),C=a()(I,3),T=C[0],D=C[1],w=C[2],L=(null==O||null===(n=O.breakpoints)||void 0===n?void 0:n.md)||728,x=(0,m.Av)(),B=(0,c.f0)(k);return o.createElement(f,{source:k},o.createElement(d.J,{ariaId:v,isVisible:T,hide:w,popoverRenderFn:function(){return o.createElement(l.mX,null,g(w))}},o.createElement(s._,{tooltipText:y||"Share",targetDistance:10},o.createElement(r.r,{"aria-controls":v,"aria-expanded":T?"true":"false","aria-label":h,onClick:function(){var e,n=null===(e=window)||void 0===e?void 0:e.innerWidth;if(x.event("shareLinkPopover.clicked",{postId:b,listId:E,source:B}),t&&n&&n<L){var a={url:t,text:i||"",title:i||""};if(navigator.canShare&&navigator.canShare(a))return void navigator.share(a)}D()},rules:"ICON_SUBTLE"===_?N.OL:N.Yq},o.createElement(p,null)))))}},57572:(e,n,t)=>{"use strict";t.d(n,{o:()=>i});var i={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"SharePostButton_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}}]}},37597:(e,n,t)=>{"use strict";t.d(n,{f:()=>k,o:()=>v});var i=t(28655),a=t.n(i),o=t(92471),l=t(67294),d=t(33384),r=t(18627),s=t(66411),m=t(43487),c=t(50458);function u(){var e=a()(["\n  fragment SharePostButton_post on Post {\n    id\n  }\n"]);return u=function(){return e},e}var k=function(e){var n,t=e.post,i=e.socialPlatform,a=e.buttonStyle,o=(0,r.Av)(),u=(0,s.Qi)(),k=(0,m.v9)((function(e){return e.config.authDomain}));if("FACEBOOK"===i)n=(0,c.VCf)(k,t.id);else if("TWITTER"===i)n=(0,c.A2M)(k,t.id);else{if("LINKEDIN"!==i)return null;n=(0,c.mZD)(k,t.id)}return l.createElement(d.T,{baseOnClick:function(){u&&o.event("post.shareOpen",{postId:t.id,source:u,dest:i.toLowerCase(),dialogType:"native"})},href:n,socialPlatform:i,buttonStyle:a})},v=(0,o.Ps)(u())},15855:(e,n,t)=>{"use strict";t.d(n,{E:()=>l});var i=t(319),a=t.n(i),o=t(57572),l={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"SharePostButtons_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"isLimitedState"}},{kind:"Field",name:{kind:"Name",value:"visibility"}},{kind:"Field",name:{kind:"Name",value:"mediumUrl"}},{kind:"FragmentSpread",name:{kind:"Name",value:"SharePostButton_post"}}]}}].concat(a()(o.o.definitions))}},75399:(e,n,t)=>{"use strict";t.d(n,{E:()=>g,U:()=>h});var i=t(67154),a=t.n(i),o=t(6479),l=t.n(o),d=t(28655),r=t.n(d),s=t(92471),m=t(67294),c=t(12287),u=t(37597),k=t(77355),v=t(30020),p=t(66411);function S(){var e=r()(["\n  fragment SharePostButtons_post on Post {\n    id\n    isLimitedState\n    visibility\n    mediumUrl\n    ...SharePostButton_post\n  }\n  ","\n"]);return S=function(){return e},e}function N(e){switch(e){case"TWITTER":return"Share on Twitter";case"FACEBOOK":return"Share on Facebook";case"LINKEDIN":return"Share on LinkedIn";default:return"Copy link"}}var f=function(e){var n=e.post,t=e.source,i=e.socialPlatform,a=e.useSubtleShareButtons,o=e.spacing,l=a?"LINK_SUBTLE":"LINK",d=a?"ICON_SUBTLE":"ICON";return n.isLimitedState&&(l="LINK_DISABLED",d="ICON_DISABLED"),m.createElement(k.x,{flexGrow:"0",paddingRight:o},"UNLISTED"!==n.visibility&&m.createElement(v._,{tooltipText:N(i),targetDistance:10},i?m.createElement(u.f,{socialPlatform:i,buttonStyle:l,post:n}):n.mediumUrl&&m.createElement(c._,{url:n.mediumUrl,copyStyle:d,source:t,reportData:{postId:n.id}})))},g=(0,s.Ps)(S(),u.o);function h(e){var n=e.spacing,t=void 0===n?"1px":n,i=l()(e,["spacing"]);return m.createElement(m.Fragment,null,m.createElement(p.cW,{source:{name:i.source}},m.createElement(f,a()({},i,{spacing:t,socialPlatform:"TWITTER"})),m.createElement(f,a()({},i,{spacing:t,socialPlatform:"FACEBOOK"})),m.createElement(f,a()({},i,{spacing:t,socialPlatform:"LINKEDIN"}))),m.createElement(f,i))}},69724:(e,n,t)=>{"use strict";t.d(n,{k:()=>i});var i={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostScrollTracker_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"collection"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"Field",name:{kind:"Name",value:"sequence"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"sequenceId"}}]}}]}}]}},68821:(e,n,t)=>{"use strict";t.d(n,{V:()=>p,k:()=>S});var i=t(28655),a=t.n(i),o=t(23493),l=t.n(o),d=t(92471),r=t(67294),s=t(18627),m=t(66411),c=t(34135),u=t(77280),k=t(84509);function v(){var e=a()(["\n  fragment PostScrollTracker_post on Post {\n    id\n    collection {\n      id\n    }\n    sequence {\n      sequenceId\n    }\n  }\n"]);return v=function(){return e},e}function p(e,n,t,i){var a=(0,s.Av)(),o=(0,m.pK)(),d=(0,u.he)(),v=(0,m.Qi)(),p=Date.now(),S=null,N=r.useCallback(l()((function(){if(e.current){var i=(0,k.L6)(e.current);if((0,k.pn)(i)){var l=Date.now(),r=(0,k.tM)(i),s=(0,k.UO)(),m=(0,k.t_)(),c=Math.round(r.top),u=Math.round(r.top+r.height),N=s.top,f=s.top+m.height,g=s.height,h={postIds:[n.id],collectionIds:[n.collection?n.collection.id:""],sequenceIds:[n.sequence?n.sequence.sequenceId:""],sources:t?["post_page"]:[v],tops:[c],bottoms:[u],areFullPosts:[!0],loggedAt:l,timeDiff:null!==S?Math.round(l-S):0,scrollTop:N,scrollBottom:f,scrollableHeight:g,viewStartedAt:p},F={referrer:d,referrerSource:o};t?a.event("post.streamScrolled",h,F):a.event("post.streamScrolled",h),S=l}}}),1e3),[n,t]);r.useEffect((function(){N();var e=null!=i&&i.current?(0,c.jC)(null==i?void 0:i.current):c.V6;return e.on("scroll_end",N),function(){e.off("scroll_end",N)}}),[N])}var S=(0,d.Ps)(v())}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/1681.06f39760.chunk.js.map