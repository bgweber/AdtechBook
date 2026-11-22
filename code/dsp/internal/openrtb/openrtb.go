// Package openrtb holds the subset of OpenRTB 2.6 that the bidder reads.
//
// Exchanges send more fields than this, and each one sends them a little
// differently. The structs keep only what the bid path uses, and every
// field is optional: a missing field decodes to its zero value, and
// normalization (package normalize) decides what a missing value means.
package openrtb

import "encoding/json"

// BidRequest is one auction, as an exchange sends it.
type BidRequest struct {
	ID     string  `json:"id"`
	Imp    []Imp   `json:"imp"`
	App    *App    `json:"app,omitempty"`
	Device *Device `json:"device,omitempty"`
	TMax   int     `json:"tmax,omitempty"` // milliseconds to respond
	AT     int     `json:"at,omitempty"`   // 1 = first price, 2 = second price
}

// Imp is one ad slot. Mobile in-app requests almost always have one.
type Imp struct {
	ID          string          `json:"id"`
	Banner      *Banner         `json:"banner,omitempty"`
	Video       *Video          `json:"video,omitempty"`
	Native      *Native         `json:"native,omitempty"`
	Instl       int             `json:"instl,omitempty"`
	Rwdd        int             `json:"rwdd,omitempty"`
	BidFloor    float64         `json:"bidfloor,omitempty"`
	BidFloorCur string          `json:"bidfloorcur,omitempty"`
	Ext         json.RawMessage `json:"ext,omitempty"`
}

type Banner struct {
	W int `json:"w,omitempty"`
	H int `json:"h,omitempty"`
}

type Video struct {
	Mimes []string `json:"mimes,omitempty"`
	W     int      `json:"w,omitempty"`
	H     int      `json:"h,omitempty"`
}

type Native struct {
	Request string `json:"request,omitempty"`
	Ver     string `json:"ver,omitempty"`
}

type App struct {
	Bundle string `json:"bundle,omitempty"`
}

type Device struct {
	IFA            string `json:"ifa,omitempty"`
	LMT            int    `json:"lmt,omitempty"`
	OS             string `json:"os,omitempty"`
	OSV            string `json:"osv,omitempty"`
	Make           string `json:"make,omitempty"`
	Model          string `json:"model,omitempty"`
	ConnectionType int    `json:"connectiontype,omitempty"`
	Geo            *Geo   `json:"geo,omitempty"`
}

type Geo struct {
	Country string `json:"country,omitempty"`
}

// BidResponse is our answer to an auction. A no-bid is an empty HTTP 204,
// so a response always has at least one bid.
type BidResponse struct {
	ID      string    `json:"id"`
	SeatBid []SeatBid `json:"seatbid"`
	Cur     string    `json:"cur,omitempty"`
}

type SeatBid struct {
	Bid []Bid `json:"bid"`
}

// Bid is one bid on one impression. The notice URLs carry OpenRTB macros
// such as ${AUCTION_PRICE}, which the exchange fills in.
type Bid struct {
	ID    string  `json:"id"`
	ImpID string  `json:"impid"`
	Price float64 `json:"price"`          // CPM, in the request's currency
	CID   string  `json:"cid,omitempty"`  // campaign ID
	CrID  string  `json:"crid,omitempty"` // creative ID
	AdM   string  `json:"adm,omitempty"`  // ad markup, with click trackers
	NURL  string  `json:"nurl,omitempty"` // win notice
	BURL  string  `json:"burl,omitempty"` // billing notice
	LURL  string  `json:"lurl,omitempty"` // loss notice
}
