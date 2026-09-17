// Package main implements evidencecc — the IBVAP evidence anchoring
// chaincode (BUILD_SPEC §7.12).
//
// It stores Merkle roots, nothing else. No evidence, no images, no plate text,
// no personal data ever touches the ledger — only a 32-byte root that proves a
// batch of alerts existed, unchanged, at a point in time. That is the whole
// design: a permissioned ledger is a good notary and a terrible database, and
// putting border surveillance data on any chain would be a privacy disaster.
//
// Anchors are immutable by construction: AnchorBatch refuses to overwrite an
// existing root, and there is no update or delete transaction at all. An
// insider who edits the alerts table cannot also edit the chain, which is what
// makes the tampering detectable (§12).
package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct {
	contractapi.Contract
}

// Anchor is one Merkle root committed to the channel.
type Anchor struct {
	MerkleRoot string `json:"merkle_root"`
	LeafCount  int    `json:"leaf_count"`
	SiteCode   string `json:"site_code"`
	AnchoredAt string `json:"anchored_at"`
	TxID       string `json:"tx_id"`
	Submitter  string `json:"submitter"`
}

// AnchorBatch writes a Merkle root. It will not overwrite one.
func (s *SmartContract) AnchorBatch(
	ctx contractapi.TransactionContextInterface,
	merkleRoot string,
	leafCount int,
	siteCode string,
	anchoredAt string,
) (*Anchor, error) {
	if len(merkleRoot) != 64 {
		return nil, fmt.Errorf(
			"merkle root must be a 64-character SHA-256 hex digest, got %d characters",
			len(merkleRoot))
	}
	if leafCount <= 0 {
		return nil, fmt.Errorf("a batch must contain at least one leaf, got %d", leafCount)
	}

	existing, err := ctx.GetStub().GetState(merkleRoot)
	if err != nil {
		return nil, fmt.Errorf("reading world state: %w", err)
	}
	if existing != nil {
		// Immutability is the point. Re-anchoring the same root is almost
		// certainly a retry after a timeout, so we fail loudly rather than
		// silently replacing a record someone may already have verified.
		return nil, fmt.Errorf("root %s is already anchored; anchors are immutable", merkleRoot)
	}

	identity, err := ctx.GetClientIdentity().GetID()
	if err != nil {
		return nil, fmt.Errorf("reading submitter identity: %w", err)
	}

	anchor := Anchor{
		MerkleRoot: merkleRoot,
		LeafCount:  leafCount,
		SiteCode:   siteCode,
		AnchoredAt: anchoredAt,
		TxID:       ctx.GetStub().GetTxID(),
		Submitter:  identity,
	}

	payload, err := json.Marshal(anchor)
	if err != nil {
		return nil, fmt.Errorf("marshalling anchor: %w", err)
	}
	if err := ctx.GetStub().PutState(merkleRoot, payload); err != nil {
		return nil, fmt.Errorf("writing anchor: %w", err)
	}
	// Indexed by transaction id too, so GetAnchor works from a receipt.
	if err := ctx.GetStub().PutState("tx~"+anchor.TxID, payload); err != nil {
		return nil, fmt.Errorf("writing tx index: %w", err)
	}
	return &anchor, nil
}

// GetAnchor fetches an anchor by transaction id.
func (s *SmartContract) GetAnchor(
	ctx contractapi.TransactionContextInterface, txID string,
) (*Anchor, error) {
	payload, err := ctx.GetStub().GetState("tx~" + txID)
	if err != nil {
		return nil, fmt.Errorf("reading world state: %w", err)
	}
	if payload == nil {
		return nil, fmt.Errorf("no anchor for transaction %s", txID)
	}
	var anchor Anchor
	if err := json.Unmarshal(payload, &anchor); err != nil {
		return nil, fmt.Errorf("unmarshalling anchor: %w", err)
	}
	return &anchor, nil
}

// VerifyRoot reports whether a root is on the chain. This is the query a
// third-party verifier makes, and it is deliberately read-only and cheap.
func (s *SmartContract) VerifyRoot(
	ctx contractapi.TransactionContextInterface, merkleRoot string,
) (bool, error) {
	payload, err := ctx.GetStub().GetState(merkleRoot)
	if err != nil {
		return false, fmt.Errorf("reading world state: %w", err)
	}
	return payload != nil, nil
}

// GetHistory returns the full immutable history for a root. On a correctly
// behaving network this has exactly one entry; more than one would itself be
// evidence of something worth investigating.
func (s *SmartContract) GetHistory(
	ctx contractapi.TransactionContextInterface, merkleRoot string,
) ([]Anchor, error) {
	iterator, err := ctx.GetStub().GetHistoryForKey(merkleRoot)
	if err != nil {
		return nil, fmt.Errorf("reading history: %w", err)
	}
	defer iterator.Close()

	var history []Anchor
	for iterator.HasNext() {
		record, err := iterator.Next()
		if err != nil {
			return nil, fmt.Errorf("iterating history: %w", err)
		}
		var anchor Anchor
		if err := json.Unmarshal(record.Value, &anchor); err == nil {
			history = append(history, anchor)
		}
	}
	return history, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		panic(fmt.Sprintf("creating evidencecc chaincode: %v", err))
	}
	if err := chaincode.Start(); err != nil {
		panic(fmt.Sprintf("starting evidencecc chaincode: %v", err))
	}
}
